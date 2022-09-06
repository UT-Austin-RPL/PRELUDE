from numpy import pi as PI

# Simple PID control handler
class PIDHandler():
    """
    A simle PID feedback controller with feedforward inputs, mainly used for joint-torque control.
    """


    def __init__(self, gainP = 0, gainD = 0, gainI = 0, gainF = 0, conFreq = 10 , conIRatioMax = 0.3, outMax = 100 , flagRotation = False):
        """
        Generate a simle PID controller.

        Args:
            gainP: proportional gain for feedback
            gainD: differential gain for feedback
            gainI: integral gain for feedback
            gainF: proportional gain for feedforward
            conFreq: control frequency [Hz] for computing integral and differential terms
            conIRatioMax: maximum ratio of integral terms with respect to PID outputs
            outMax: maximum value of PID outputs
            flagRot: flag for rotational-state feedback

        Returns:
            None
        """

        self.gainP = gainP
        self.gainI = gainI
        self.gainD = gainD

        self.gainF = gainF

        self.conFreq = conFreq
        self.conPeriod = 1.0/conFreq

        self.flagRot = flagRotation

        if gainI > 0:
            self.conIMax = conIRatioMax /(self.gainI * self.conPeriod)
        else:
            self.conIMax = 0

        self.coffLPF = 0.7

        self.outMax = outMax
        self.outMin = - outMax

        self.reset()

        return


    def reset(self):
        """
        Reset the controller.

        Args:
            None

        Returns:
            None
        """

        self._fwdCur = 0
        self._fdbCur = 0
        self._fdbPrv = 0
        self._fdbDif = 0
        self._fdbSum = 0

        self._out = 0

        return


    def update(self, fdbVal = 0, fwdVal = 0):
        """
        Update the controller with given feedback/feedforward values.

        Args:
            fdbVal: state for feedback control
            fwdVal: command for feedforward control

        Returns:
            None
        """

        self._fwdCur = fwdVal
        self._fdbCur = fdbVal

        if self.flagRot:
            while self._fdbCur > PI:
                self._fdbCur -= PI
            while self._fdbCur < - PI:
                self._fdbCur += PI

        if self._fwdCur ==  0:
            self._fdbDif = self.coffLPF * (self._fdbCur - self._fdbPrv) + (1 - self.coffLPF) * self._fdbDif 
            self._fdbSum += self._fdbCur

            if self._fdbSum > self.conIMax:
                self._fdbSum = self.conIMax
            elif self._fdbSum < - self.conIMax:
                self._fdbSum = - self.conIMax
        else:
            self._fdbDif = 0
            self._fdbSum = 0

        self._fdbPrv = self._fdbCur

        uF = self.gainF * self._fwdCur
        uP = self.gainP * self._fdbCur
        uD = self.gainD * self.conFreq * self._fdbDif
        uI = self.gainI * self.conPeriod * self._fdbSum

        self._out = uP + uD + uI + uF
        if self._out > self.outMax:
            self._out = self.outMax
        elif self._out < self.outMin:
            self._out = self.outMin

        return


    def output(self):
        """
        Output control values.

        Args:
            None

        Returns:
            Control output (float-type, equlvalent to self._out)
        """

        return self._out
