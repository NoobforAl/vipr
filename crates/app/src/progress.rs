//! The bar shown while work runs. The filled part is the real progress, eased so it glides instead of jumping. A soft
//! pulse moves through the rest, so a long step with nothing to report (loading a model) doesn't look frozen. The pulse
//! says nothing about how far along the work is, so it never adds to the filled part.

use iced::widget::{container, row, space};
use iced::{Element, Length};

use crate::theme::{self, Tokens};

/// Share of the remaining distance covered per animation tick.
const EASING: f32 = 0.14;
/// Closer than this, the bar just arrives.
const SNAP: f32 = 0.001;
/// Seconds for the pulse to cross the bar once.
const PULSE_SECONDS: f32 = 1.6;
/// Width of the pulse, in `PULSE_SCALE` parts of the unfilled space.
const PULSE_WIDTH: f32 = 300.0;
const PULSE_SCALE: f32 = 1000.0;
/// Parts of the whole bar.
const BAR_SCALE: f32 = 10_000.0;

/// The next value to show on the way to `target`. Starting a new run goes straight back, without a slide.
pub fn approach(shown: f32, target: f32) -> f32 {
    if target <= shown || target - shown < SNAP {
        return target;
    }
    shown + (target - shown) * EASING
}

/// Where the pulse is at `clock` seconds: the portions before it, of it and after it, as parts of the unfilled space.
/// It slides in from the left edge and out at the right.
fn pulse(clock: f32) -> (u16, u16, u16) {
    let t = (clock / PULSE_SECONDS).rem_euclid(1.0);
    let eased = t * t * (3.0 - 2.0 * t);
    let left = -PULSE_WIDTH + (PULSE_SCALE + PULSE_WIDTH) * eased;
    let start = left.max(0.0);
    let end = (left + PULSE_WIDTH).min(PULSE_SCALE);
    let before = start.round() as u16;
    let size = (end - start).max(0.0).round() as u16;
    let scale = PULSE_SCALE as u16;
    (before, size, scale.saturating_sub(before + size))
}

/// How the bar divides between filled and unfilled, as portions.
fn split(fraction: f32) -> (u16, u16) {
    let filled = (fraction.clamp(0.0, 1.0) * BAR_SCALE).round() as u16;
    (filled, BAR_SCALE as u16 - filled)
}

fn gap<'a, M: 'a>(portion: u16) -> Option<Element<'a, M>> {
    (portion > 0).then(|| space().width(Length::FillPortion(portion)).into())
}

/// A bar `height` px tall, `fraction` (0..1) filled. `clock` is any steady time in seconds.
pub fn bar<'a, M: 'a>(fraction: f32, clock: f32, height: f32, t: Tokens) -> Element<'a, M> {
    let radius = height / 2.0;
    let (filled, open) = split(fraction);
    let mut line = row![].height(height).width(Length::Fill);
    if filled > 0 {
        line = line.push(
            container(space())
                .width(Length::FillPortion(filled))
                .height(height)
                .style(theme::rounded_fill(t.accent, radius)),
        );
    }
    if open > 0 {
        let (before, size, after) = pulse(clock);
        let mut travel = row![].height(height).width(Length::Fill).push(gap(before));
        if size > 0 {
            travel = travel.push(
                container(space())
                    .width(Length::FillPortion(size))
                    .height(height)
                    .style(theme::rounded_fill(
                        theme::with_alpha(t.accent, 0.38),
                        radius,
                    )),
            );
        }
        line = line.push(
            container(travel.push(gap(after)))
                .width(Length::FillPortion(open))
                .height(height),
        );
    }
    container(line)
        .width(Length::Fill)
        .height(height)
        .clip(true)
        .style(theme::rounded_fill(t.rule, radius))
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shown_value_glides_up_and_arrives() {
        let mut shown = 0.0;
        let mut last = shown;
        for _ in 0..200 {
            shown = approach(shown, 0.5);
            assert!(shown >= last && shown <= 0.5);
            last = shown;
        }
        assert_eq!(shown, 0.5);
    }

    #[test]
    fn it_never_shows_more_than_the_real_progress() {
        let mut shown = 0.0;
        for target in [0.1, 0.1, 0.4, 0.4, 0.9] {
            shown = approach(shown, target);
            assert!(shown <= target);
        }
    }

    #[test]
    fn a_new_run_starts_from_the_start_at_once() {
        assert_eq!(approach(0.8, 0.0), 0.0);
    }

    #[test]
    fn the_pulse_always_fills_its_track_exactly_and_keeps_moving() {
        let mut seen = std::collections::HashSet::new();
        for step in 0..400 {
            let (before, size, after) = pulse(step as f32 * 0.033);
            assert_eq!(before + size + after, PULSE_SCALE as u16, "at step {step}");
            seen.insert(before);
        }
        assert!(seen.len() > 20, "it should travel, not sit still");
    }

    #[test]
    fn the_pulse_handles_any_clock() {
        for clock in [0.0, -5.0, 1e6] {
            let (before, size, after) = pulse(clock);
            assert_eq!(before + size + after, PULSE_SCALE as u16);
        }
    }

    #[test]
    fn the_bar_splits_cleanly_at_the_ends_and_out_of_range() {
        assert_eq!(split(0.0), (0, 10_000));
        assert_eq!(split(1.0), (10_000, 0));
        assert_eq!(split(0.25), (2500, 7500));
        assert_eq!(split(-1.0), (0, 10_000));
        assert_eq!(split(7.0), (10_000, 0));
    }
}
