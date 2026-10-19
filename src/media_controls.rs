//! macOS media integration, compiled with the `macos_media_controls` feature.
//!
//! Inside the spotifyd process this
//!   * mirrors playback into the Now Playing widget (`MPNowPlayingInfoCenter`),
//!   * handles media keys / headset buttons (`MPRemoteCommandCenter`) by driving `Spirc`,
//!   * keeps a now-playing JSON file up to date for shell prompts and other tools.
//!
//! Apple's media APIs need an AppKit run loop on the main thread, so `main` runs the tokio
//! runtime on a worker thread and hands the main thread to [`run_main_thread`]. The main loop
//! talks to this module only through [`MediaHandle`]: one `on_event` tee next to the MPRIS one,
//! and `set_spirc` when a connection comes and goes.
//!
//! Configuration is by environment, to keep the footprint in `config.rs` at zero:
//!   * `SPOTIFYD_MEDIA_CONTROLS=0` disables the feature at runtime;
//!   * `SPOTIFYD_NOW_PLAYING_FILE=<path>` overrides the JSON file
//!     (default `$XDG_STATE_HOME/spotifyd/now_playing.json`, i.e. `~/.local/state/spotifyd/...`).

use std::{
    cell::RefCell,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use color_eyre::eyre::{self, Context as _};
use librespot_connect::Spirc;
use librespot_metadata::audio::{AudioItem, UniqueFields};
use librespot_playback::player::PlayerEvent;
use log::{debug, info, warn};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSEvent, NSEventModifierFlags, NSEventSubtype,
    NSEventType,
};
use objc2_foundation::{MainThreadMarker, NSPoint};
use souvlaki::{
    MediaControlEvent, MediaControls, MediaMetadata, MediaPlayback, MediaPosition, PlatformConfig,
};

/// Whether the integration should run (`SPOTIFYD_MEDIA_CONTROLS`, default on).
pub(crate) fn enabled() -> bool {
    !matches!(
        std::env::var("SPOTIFYD_MEDIA_CONTROLS").as_deref(),
        Ok("0") | Ok("false") | Ok("no") | Ok("off")
    )
}

fn now_playing_file() -> PathBuf {
    if let Some(path) = std::env::var_os("SPOTIFYD_NOW_PLAYING_FILE") {
        return PathBuf::from(path);
    }
    let state_home = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(std::env::temp_dir);
    state_home.join("spotifyd").join("now_playing.json")
}

/// The main loop's end of the integration. Cheap to clone; dropping the last clone (i.e. the
/// main loop finishing) makes [`run_main_thread`] return.
#[derive(Clone)]
pub(crate) struct MediaHandle {
    events: Sender<PlayerEvent>,
    spirc: Arc<Mutex<Option<Arc<Spirc>>>>,
}

impl MediaHandle {
    pub(crate) fn on_event(&self, event: &PlayerEvent) {
        let _ = self.events.send(event.clone());
    }

    /// The connect handle of the current session, or `None` between sessions.
    pub(crate) fn set_spirc(&self, spirc: Option<Arc<Spirc>>) {
        *self.spirc.lock().unwrap_or_else(|e| e.into_inner()) = spirc;
    }
}

/// Creates the handle for the main loop and the closure that owns the main thread.
/// `device_name` labels the Now Playing tile while nothing is playing here.
pub(crate) fn new(device_name: String) -> (MediaHandle, impl FnOnce() -> eyre::Result<()>) {
    let (tx, rx) = mpsc::channel();
    let spirc = Arc::new(Mutex::new(None));
    let handle = MediaHandle {
        events: tx,
        spirc: spirc.clone(),
    };
    (handle, move || run_main_thread(rx, spirc, device_name))
}

// ---------------------------------------------------------------------------------------------
// Playback state, mirrored from the player events (same transitions as dbus_mpris).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Playing,
    Paused,
    Stopped,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Playing => "playing",
            Status::Paused => "paused",
            Status::Stopped => "stopped",
        }
    }
}

#[derive(Debug, Default)]
struct State {
    status: Option<Status>,
    item: Option<Box<AudioItem>>,
    /// Position at `position_at`; extrapolated while playing.
    position: Duration,
    position_at: Option<Instant>,
    /// Whether this device is the active Spotify Connect device.
    active: bool,
}

impl State {
    fn status(&self) -> Status {
        self.status.unwrap_or(Status::Stopped)
    }

    fn position(&self) -> Duration {
        match (self.status(), self.position_at) {
            (Status::Playing, Some(at)) => self.position + at.elapsed(),
            _ => self.position,
        }
    }

    fn set_position(&mut self, position_ms: u32) {
        self.position = Duration::from_millis(position_ms.into());
        self.position_at = Some(Instant::now());
    }

    /// Applies an event; returns whether anything user-visible changed.
    fn apply(&mut self, event: PlayerEvent) -> bool {
        match event {
            PlayerEvent::TrackChanged { audio_item } => {
                self.item = Some(audio_item);
                true
            }
            PlayerEvent::Playing { position_ms, .. } => {
                self.status = Some(Status::Playing);
                self.set_position(position_ms);
                true
            }
            PlayerEvent::Paused { position_ms, .. } => {
                self.status = Some(Status::Paused);
                self.set_position(position_ms);
                true
            }
            PlayerEvent::Stopped { .. } => {
                self.status = Some(Status::Stopped);
                self.item = None;
                self.set_position(0);
                true
            }
            PlayerEvent::Seeked { position_ms, .. }
            | PlayerEvent::PositionCorrection { position_ms, .. }
            | PlayerEvent::PositionChanged { position_ms, .. } => {
                self.set_position(position_ms);
                true
            }
            PlayerEvent::SessionConnected { .. } => {
                self.active = true;
                false
            }
            PlayerEvent::SessionDisconnected { .. } => {
                self.active = false;
                self.status = Some(Status::Stopped);
                self.item = None;
                self.set_position(0);
                true
            }
            _ => false,
        }
    }

    fn artists(&self) -> Vec<String> {
        match self.item.as_deref().map(|item| &item.unique_fields) {
            Some(UniqueFields::Track { artists, .. }) => {
                artists.0.iter().map(|a| a.name.clone()).collect()
            }
            Some(UniqueFields::Local { artists, .. }) => artists.iter().cloned().collect(),
            Some(UniqueFields::Episode { show_name, .. }) => vec![show_name.clone()],
            None => vec![],
        }
    }

    fn album(&self) -> Option<String> {
        match self.item.as_deref().map(|item| &item.unique_fields) {
            Some(UniqueFields::Track { album, .. }) => Some(album.clone()),
            Some(UniqueFields::Local { album, .. }) => album.clone(),
            _ => None,
        }
    }

    fn cover_url(&self) -> Option<&str> {
        self.item
            .as_deref()?
            .covers
            .iter()
            .max_by_key(|c| c.width)
            .map(|c| c.url.as_str())
    }

    fn playback(&self) -> MediaPlayback {
        let progress = Some(MediaPosition(self.position()));
        match self.status() {
            Status::Playing => MediaPlayback::Playing { progress },
            Status::Paused => MediaPlayback::Paused { progress },
            // Media keys keep arriving while stopped, so playing from here still takes over.
            Status::Stopped => MediaPlayback::Stopped,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let mut json = serde_json::json!({
            "status": self.status().as_str(),
            "updated_at": updated_at,
        });
        if let Some(item) = self
            .item
            .as_deref()
            .filter(|_| self.status() != Status::Stopped)
        {
            json["track_id"] = item.track_id.to_uri().unwrap_or_default().into();
            json["title"] = item.name.clone().into();
            json["artists"] = self.artists().into();
            json["album"] = self.album().into();
            json["duration_ms"] = item.duration_ms.into();
            json["position_ms"] = (self.position().as_millis() as u64).into();
            json["cover_url"] = self.cover_url().into();
        }
        json
    }
}

// ---------------------------------------------------------------------------------------------
// Main thread: AppKit run loop, Now Playing, media keys, the file.

/// Artwork for the idle tile (a music note on a dark square); macOS needs it as a file.
const IDLE_ARTWORK: &[u8] = include_bytes!("media_controls/idle_artwork.png");

struct Context {
    controls: MediaControls,
    state: State,
    file: PathBuf,
    device_name: String,
    idle_artwork_url: Option<String>,
}

thread_local! {
    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}

fn with_context(f: impl FnOnce(&mut Context)) {
    CONTEXT.with(|ctx| {
        if let Ok(mut ctx) = ctx.try_borrow_mut() {
            if let Some(ctx) = ctx.as_mut() {
                f(ctx);
            }
        } else {
            warn!("media controls context busy; dropping update");
        }
    });
}

impl Context {
    fn publish(&mut self) {
        let state = &self.state;
        let artists = state.artists().join(", ");
        let album = state.album();
        let item = state
            .item
            .as_deref()
            .filter(|_| state.status() != Status::Stopped);
        debug!(
            "now playing: {:?} {:?}",
            state.status(),
            item.map(|i| &i.name)
        );
        if let Err(err) = self.controls.set_playback(state.playback()) {
            warn!("failed to update Now Playing playback state: {err:?}");
        }
        // While stopped (nothing played yet, or another device took playback over) the tile
        // stays: macOS keeps every process that registered media-key handlers in Now Playing,
        // and those handlers are what lets a play press take playback back. Without metadata
        // it would show the process name, so label it instead.
        let metadata = match item {
            Some(item) => MediaMetadata {
                title: Some(item.name.as_str()),
                artist: Some(artists.as_str()),
                album: album.as_deref(),
                cover_url: state.cover_url(),
                duration: Some(Duration::from_millis(item.duration_ms.into())),
            },
            None => MediaMetadata {
                title: Some("Not playing"),
                artist: Some(self.device_name.as_str()),
                cover_url: self.idle_artwork_url.as_deref(),
                ..MediaMetadata::default()
            },
        };
        if let Err(err) = self.controls.set_metadata(metadata) {
            warn!("failed to update Now Playing metadata: {err:?}");
        }
        if let Err(err) = write_atomically(&self.file, &state.to_json()) {
            warn!("failed to write {}: {err}", self.file.display());
        }
    }
}

/// Puts the idle artwork next to the now-playing file and returns its `file://` URL.
fn write_idle_artwork(now_playing_file: &Path) -> eyre::Result<String> {
    let dir = now_playing_file
        .parent()
        .ok_or_else(|| eyre::eyre!("now-playing file has no parent directory"))?;
    fs::create_dir_all(dir)?;
    let path = dir.join("idle_artwork.png");
    if fs::read(&path).ok().as_deref() != Some(IDLE_ARTWORK) {
        fs::write(&path, IDLE_ARTWORK)?;
    }
    Ok(format!("file://{}", path.display()))
}

fn write_atomically(path: &Path, json: &serde_json::Value) -> eyre::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut file, json)?;
        file.write_all(b"\n")?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Media key handler. Runs on the main thread, invoked by `MPRemoteCommandCenter`.
fn handle_media_event(spirc: &Mutex<Option<Arc<Spirc>>>, event: MediaControlEvent) {
    let guard = spirc.lock().unwrap_or_else(|e| e.into_inner());
    let Some(spirc) = guard.as_ref() else {
        debug!("media key {event:?} ignored: not connected");
        return;
    };
    let mut active = false;
    with_context(|ctx| active = ctx.state.active);

    let result = match &event {
        // Play on an inactive device takes playback over from whatever is playing elsewhere,
        // like picking this device in a Spotify client.
        MediaControlEvent::Play | MediaControlEvent::Toggle if !active => {
            info!("media key {event:?}: taking over playback");
            spirc.transfer(None)
        }
        MediaControlEvent::Play => spirc.play(),
        MediaControlEvent::Pause => spirc.pause(),
        MediaControlEvent::Toggle => spirc.play_pause(),
        MediaControlEvent::Next => spirc.next(),
        MediaControlEvent::Previous => spirc.prev(),
        MediaControlEvent::Stop => spirc.pause(),
        MediaControlEvent::SetPosition(MediaPosition(pos)) => {
            spirc.set_position_ms(pos.as_millis().min(u32::MAX as u128) as u32)
        }
        MediaControlEvent::Seek(direction) => {
            let mut target = Duration::ZERO;
            with_context(|ctx| target = ctx.state.position());
            let step = Duration::from_secs(10);
            let target = match direction {
                souvlaki::SeekDirection::Forward => target + step,
                souvlaki::SeekDirection::Backward => target.saturating_sub(step),
            };
            spirc.set_position_ms(target.as_millis().min(u32::MAX as u128) as u32)
        }
        other => {
            debug!("media key {other:?} not handled");
            Ok(())
        }
    };
    if let Err(err) = result {
        warn!("media key {event:?} failed: {err}");
    }
}

/// Owns the main thread until the main loop drops its [`MediaHandle`].
fn run_main_thread(
    events: Receiver<PlayerEvent>,
    spirc: Arc<Mutex<Option<Arc<Spirc>>>>,
    device_name: String,
) -> eyre::Result<()> {
    let mtm = MainThreadMarker::new()
        .ok_or_else(|| eyre::eyre!("media controls must be started on the main thread"))?;
    let app = NSApplication::sharedApplication(mtm);
    // No Dock icon, no menu bar, no activation.
    app.setActivationPolicy(NSApplicationActivationPolicy::Prohibited);

    let mut controls = MediaControls::new(PlatformConfig {
        dbus_name: "spotifyd",
        display_name: "spotifyd",
        hwnd: None,
    })
    .map_err(|e| eyre::eyre!("failed to create media controls: {e:?}"))?;
    controls
        .attach(move |event| handle_media_event(&spirc, event))
        .map_err(|e| eyre::eyre!("failed to attach media key handler: {e:?}"))?;
    // macOS only starts showing metadata once the app has reported itself as playing.
    controls
        .set_playback(MediaPlayback::Playing { progress: None })
        .map_err(|e| eyre::eyre!("failed to initialise Now Playing: {e:?}"))?;

    let file = now_playing_file();
    info!(
        "media controls enabled; now-playing file at {}",
        file.display()
    );
    let idle_artwork_url = write_idle_artwork(&file)
        .map_err(|err| warn!("no idle artwork: {err}"))
        .ok();
    CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(Context {
            controls,
            state: State::default(),
            file,
            device_name,
            idle_artwork_url,
        })
    });
    with_context(|ctx| ctx.publish());

    // Pump: forward player events onto the main queue; stop the app when the sender is gone.
    std::thread::Builder::new()
        .name("media-events".into())
        .spawn(move || {
            let main = dispatch2::DispatchQueue::main();
            while let Ok(event) = events.recv() {
                main.exec_async(move || {
                    with_context(|ctx| {
                        if ctx.state.apply(event) {
                            ctx.publish();
                        }
                    })
                });
            }
            main.exec_async(stop_app);
        })
        .wrap_err("failed to spawn media event pump")?;

    app.run();

    CONTEXT.with(|ctx| ctx.borrow_mut().take());
    Ok(())
}

/// `NSApplication::stop` only takes effect once the run loop sees an event, so post one.
fn stop_app() {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    app.stop(None);
    let event = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
        NSEventType::ApplicationDefined,
        NSPoint::ZERO,
        NSEventModifierFlags::empty(),
        0.0,
        0,
        None,
        NSEventSubtype::ApplicationActivated.0,
        0,
        0,
    );
    if let Some(event) = event {
        app.postEvent_atStart(&event, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopped_json_has_no_track() {
        let state = State::default();
        let json = state.to_json();
        assert_eq!(json["status"], "stopped");
        assert!(json.get("title").is_none());
    }

    #[test]
    fn session_tracks_active_flag() {
        let mut state = State::default();
        assert!(!state.active);
        state.apply(PlayerEvent::SessionConnected {
            connection_id: "c".into(),
            user_name: "u".into(),
        });
        assert!(state.active);
        state.apply(PlayerEvent::SessionDisconnected {
            connection_id: "c".into(),
            user_name: "u".into(),
        });
        assert!(!state.active);
        assert_eq!(state.status(), Status::Stopped);
    }

    #[test]
    fn position_extrapolates_only_while_playing() {
        let mut state = State::default();
        state.apply(PlayerEvent::Paused {
            play_request_id: 1,
            track_id: librespot_core::SpotifyUri::from_uri("spotify:track:4uLU6hMCjMI75M1A2tKUQC")
                .unwrap(),
            position_ms: 5_000,
        });
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(state.position(), Duration::from_millis(5_000));
        state.apply(PlayerEvent::Playing {
            play_request_id: 1,
            track_id: librespot_core::SpotifyUri::from_uri("spotify:track:4uLU6hMCjMI75M1A2tKUQC")
                .unwrap(),
            position_ms: 5_000,
        });
        std::thread::sleep(Duration::from_millis(20));
        assert!(state.position() > Duration::from_millis(5_000));
    }
}
