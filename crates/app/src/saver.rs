//! One background thread for small saves made in reaction to a click.
//!
//! A save flushes to disk, and in a synced folder (OneDrive) or on a busy disk that can take seconds.
//! On the UI thread that freezes the whole window, so these saves run here instead. One thread keeps
//! them in order: the last state saved is always the last one asked for.

use std::sync::OnceLock;
use std::sync::mpsc::{self, Sender};

type Job = Box<dyn FnOnce() + Send>;

fn sender() -> &'static Sender<Job> {
    static TX: OnceLock<Sender<Job>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::spawn(move || {
            for job in rx {
                job();
            }
        });
        tx
    })
}

/// Run `job` after every job already queued.
pub fn run(job: impl FnOnce() + Send + 'static) {
    if let Err(failed) = sender().send(Box::new(job)) {
        // the thread is gone, and losing the user's data is worse than a slow click
        (failed.0)();
    }
}

/// Block until everything queued so far has finished. For reading back what was saved, and for
/// closing the window, when losing a save would be worse than waiting for it.
pub fn wait() {
    let (tx, rx) = mpsc::channel();
    run(move || {
        let _ = tx.send(());
    });
    let _ = rx.recv();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn jobs_run_in_order_and_wait_sees_them_all() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        for i in 0..50 {
            let seen = seen.clone();
            run(move || seen.lock().unwrap().push(i));
        }
        wait();
        assert_eq!(*seen.lock().unwrap(), (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn a_job_does_not_run_on_the_calling_thread() {
        let here = std::thread::current().id();
        let there = Arc::new(Mutex::new(None));
        let slot = there.clone();
        run(move || *slot.lock().unwrap() = Some(std::thread::current().id()));
        wait();
        assert_ne!(there.lock().unwrap().unwrap(), here);
    }
}
