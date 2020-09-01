import time

class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        #self.current_time = time.time()
        #self.last_time = self.current_time
        self.clear()

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.ITerm_s = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 20.0
        self.output = 0.0
        self.timer = 0

    def update(self, feedback_error, feedback_delta_error):
        self.timer += 1
        error = feedback_error
        #self.current_time = time.time()
        #delta_time = self.current_time - self.last_time
        delta_error = feedback_delta_error
        #if (delta_time >= self.sample_time):
        self.PTerm = self.Kp * error
        self.ITerm += error # * delta_time
        if (self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif (self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard
        #if delta_time > 0:
        self.DTerm = delta_error# / delta_time
        #self.last_time = self.current_time
        #self.output = self.PTerm + (self.Ki * self.ITerm) / self.timer + (self.Kd * self.DTerm) # why /self.timer??
        self.output = self.PTerm + (self.Ki * self.ITerm)  + (self.Kd * self.DTerm)
        return self.output

    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time