#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>

int main() {
    CPU_TIMER_INIT(work2);

    // Simulated other work
    volatile double work_val = 0.0;
    for (int i = 0; i < 5'000'000; ++i) {
        work_val += rand() % 100; 
    }

    CPU_TIMER_STOP(work2);
    TIMER_PRINT(work2);

    return 0;
}