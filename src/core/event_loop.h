#ifndef _SRC_CORE_EVENT_LOOP_H_
#define _SRC_CORE_EVENT_LOOP_H_

/*
 * A minimal main-thread event loop replacing what cisTEM used from wxAppConsole: a queue of
 * callbacks that other threads post with CallAfter() (wxEvtHandler::CallAfter / wxQueueEvent),
 * one-shot timers (wxTimer::StartOnce) and ExitMainLoop().
 *
 * Run() executes callbacks and due timers on the calling thread, in order, and returns when
 * ExitMainLoop() has been called and the callbacks queued before it have run.
 */

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class EventLoop {
  public:
    EventLoop( );
    virtual ~EventLoop( );

    EventLoop(const EventLoop&) = delete;
    EventLoop& operator=(const EventLoop&) = delete;

    /// Queue a callback to run on the loop thread. Safe to call from any thread, including from a callback.
    void CallAfter(std::function<void( )> callback);

    /// Run `callback` once, `milliseconds` from now, on the loop thread. Returns a timer id for StopTimer().
    int StartOnceTimer(long milliseconds, std::function<void( )> callback);

    /// Cancel a pending timer; harmless if it already fired.
    void StopTimer(int timer_id);

    bool TimerIsPending(int timer_id) const;

    /// Process events until ExitMainLoop() is called. Returns the exit code.
    int Run( );

    /// Ask Run() to return once the queued callbacks have been processed. Safe from any thread.
    void ExitMainLoop(int exit_code = 0);

    bool ExitWasRequested( ) const;

    /// True when called on the thread currently executing Run().
    bool IsLoopThread( ) const;

  private:
    struct PendingTimer {
        int                                   id;
        std::chrono::steady_clock::time_point due;
        std::function<void( )>                callback;
    };

    mutable std::mutex                 mutex_;
    std::condition_variable            condition_;
    std::deque<std::function<void( )>> callbacks_;
    std::vector<PendingTimer>          timers_;
    int                                next_timer_id_;
    bool                               exit_requested_;
    int                                exit_code_;
    std::thread::id                    loop_thread_id_;
    bool                               running_;
};

/**
 * A thread-safe FIFO with timed receive (wxMessageQueue).
 */
template <typename T>
class MessageQueue {
  public:
    void Post(const T& message) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(message);
        }
        condition_.notify_one( );
    }

    /// Wait up to `milliseconds` for a message. Returns false on timeout.
    bool ReceiveTimeout(long milliseconds, T& message) {
        std::unique_lock<std::mutex> lock(mutex_);
        if ( ! condition_.wait_for(lock, std::chrono::milliseconds(milliseconds), [this] { return ! queue_.empty( ); }) )
            return false;
        message = queue_.front( );
        queue_.pop_front( );
        return true;
    }

    void Receive(T& message) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return ! queue_.empty( ); });
        message = queue_.front( );
        queue_.pop_front( );
    }

    void Clear( ) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear( );
    }

    size_t Size( ) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size( );
    }

  private:
    mutable std::mutex      mutex_;
    std::condition_variable condition_;
    std::deque<T>           queue_;
};

#endif // _SRC_CORE_EVENT_LOOP_H_
