#include "FixedRateExecutor.h"

FixedRateExecutor::FixedRateExecutor(unsigned long delayMicros, CallbackFunction func)
    : delayMicros(delayMicros), func(func), lastExecutionMicros(0) {
}

void FixedRateExecutor::tick(unsigned long nowMicros) {
    if (nowMicros - lastExecutionMicros < delayMicros) {
        return;
    }
    if (func) {  // Check if function is valid
        func();
    }
    lastExecutionMicros = nowMicros;

}