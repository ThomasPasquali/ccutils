#pragma once
// Functions implemented in other translation units. They never see the section or
// the timers that main() creates; they reach them through ccutils by name.
double solve_phase(int iterations);
void report_phase();
