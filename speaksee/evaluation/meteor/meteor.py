#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0,len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        raw = self.meteor_p.stdout.readline().decode().strip()
        numbers = [str(int(float(n))) for n in raw.split()]
        return ' '.join(numbers)

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
