#include "event_loop.h"

#include <algorithm>

EventLoop::EventLoop( ) : next_timer_id_(1), exit_requested_(false), exit_code_(0), running_(false) {
}

EventLoop::~EventLoop( ) {
}

void EventLoop::CallAfter(std::function<void( )> callback) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.push_back(std::move(callback));
    }
    condition_.notify_one( );
}

int EventLoop::StartOnceTimer(long milliseconds, std::function<void( )> callback) {
    int id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id = next_timer_id_++;
        timers_.push_back({id, std::chrono::steady_clock::now( ) + std::chrono::milliseconds(milliseconds), std::move(callback)});
    }
    condition_.notify_one( );
    return id;
}

void EventLoop::StopTimer(int timer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    timers_.erase(std::remove_if(timers_.begin( ), timers_.end( ), [timer_id](const PendingTimer& timer) { return timer.id == timer_id; }), timers_.end( ));
}

bool EventLoop::TimerIsPending(int timer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for ( const PendingTimer& timer : timers_ )
        if ( timer.id == timer_id )
            return true;
    return false;
}

void EventLoop::ExitMainLoop(int exit_code) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        exit_requested_ = true;
        exit_code_      = exit_code;
    }
    condition_.notify_one( );
}

bool EventLoop::ExitWasRequested( ) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exit_requested_;
}

bool EventLoop::IsLoopThread( ) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_ && std::this_thread::get_id( ) == loop_thread_id_;
}

int EventLoop::Run( ) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        loop_thread_id_ = std::this_thread::get_id( );
        running_        = true;
    }

    while ( true ) {
        std::function<void( )> next;
        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Move due timers onto the callback queue, earliest first.
            auto now = std::chrono::steady_clock::now( );
            std::sort(timers_.begin( ), timers_.end( ), [](const PendingTimer& a, const PendingTimer& b) { return a.due < b.due; });
            while ( ! timers_.empty( ) && timers_.front( ).due <= now ) {
                callbacks_.push_back(std::move(timers_.front( ).callback));
                timers_.erase(timers_.begin( ));
            }

            if ( callbacks_.empty( ) ) {
                if ( exit_requested_ )
                    break;
                if ( timers_.empty( ) )
                    condition_.wait(lock);
                else
                    condition_.wait_until(lock, timers_.front( ).due);
                continue;
            }

            next = std::move(callbacks_.front( ));
            callbacks_.pop_front( );
        }
        next( );
    }

    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    return exit_code_;
}
