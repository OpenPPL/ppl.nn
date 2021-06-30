#include "ppl/kernel/x86/common/timer.h"

namespace ppl { namespace kernel { namespace x86 {

void thread_timer_t::init(const int32_t num_timers)
{
    thread_timers_.resize(PPL_OMP_MAX_THREADS());
    for (int32_t i = 0; i < PPL_OMP_MAX_THREADS(); ++i) {
        thread_timers_[i].init(num_timers);
    }
}

void thread_timer_t::clear()
{
    for (int32_t i = 0; i < PPL_OMP_MAX_THREADS(); ++i) {
        thread_timers_[i].clear();
    }
}

void thread_timer_t::tic(const int32_t id)
{
    thread_timers_[PPL_OMP_THREAD_ID()].tic(id);
}

void thread_timer_t::toc(const int32_t id)
{
    thread_timers_[PPL_OMP_THREAD_ID()].toc(id);
}

double thread_timer_t::Seconds(const int32_t id) const
{
    return thread_timers_[PPL_OMP_THREAD_ID()].Seconds(id);
}

double thread_timer_t::Milliseconds(const int32_t id) const
{
    return thread_timers_[PPL_OMP_THREAD_ID()].Milliseconds(id);
}

double thread_timer_t::Microseconds(const int32_t id) const
{
    return thread_timers_[PPL_OMP_THREAD_ID()].Microseconds(id);
}

std::vector<double> thread_timer_t::gather_seconds(const int32_t id) const
{
    std::vector<double> ret;
    ret.resize(PPL_OMP_MAX_THREADS());
    for (int32_t i = 0; i < PPL_OMP_MAX_THREADS(); ++i) {
        ret[i] = thread_timers_[i].Seconds(id);
    }
    return ret;
}

std::vector<double> thread_timer_t::gather_milliseconds(const int32_t id) const
{
    std::vector<double> ret;
    ret.resize(PPL_OMP_MAX_THREADS());
    for (int32_t i = 0; i < PPL_OMP_MAX_THREADS(); ++i) {
        ret[i] = thread_timers_[i].Milliseconds(id);
    }
    return ret;
}

std::vector<double> thread_timer_t::gather_microseconds(const int32_t id) const
{
    std::vector<double> ret;
    ret.resize(PPL_OMP_MAX_THREADS());
    for (int32_t i = 0; i < PPL_OMP_MAX_THREADS(); ++i) {
        ret[i] = thread_timers_[i].Microseconds(id);
    }
    return ret;
}

std::string thread_timer_t::export_csv(const char **headers, const bool percentage) const
{
    std::string ret;
    if (headers) {
        for (int32_t i = 0; i < num_timers(); ++i) {
            ret.append(headers[i]);
            if (i < num_timers() - 1)
                ret.append(1, ',');
        }
        ret.append(1, '\n');
    }

    double tot_time;
    if (percentage) {
        tot_time = 0.0;
        for (int32_t t = 0; t < PPL_OMP_MAX_THREADS(); ++t) {
            for (int32_t i = 0; i < num_timers(); ++i) {
                tot_time += thread_timers_[t].Milliseconds(i);
            }
        }
        tot_time /= 100.0;
    } else {
        tot_time = 1.0;
    }

    char buf[512];
    for (int32_t t = 0; t < PPL_OMP_MAX_THREADS(); ++t) {
        for (int32_t i = 0; i < num_timers(); ++i) {
            sprintf(buf, "%.2f", thread_timers_[t].Milliseconds(i) / tot_time);
            ret.append(buf);
            if (i < num_timers() - 1)
                ret.append(1, ',');
        }
        ret.append(1, '\n');
    }

    return ret;
}

}}}; // namespace ppl::kernel::x86
