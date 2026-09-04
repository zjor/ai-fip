#ifndef FIXED_RATE_EXECUTOR_H
#define FIXED_RATE_EXECUTOR_H

typedef void (*CallbackFunction)();

class FixedRateExecutor {
private:
    unsigned long delayMicros;
    CallbackFunction func;
    unsigned long lastExecutionMicros;

public:
    FixedRateExecutor(unsigned long delayMicros, CallbackFunction func);    
    void tick(unsigned long nowMicros);    
};

#endif