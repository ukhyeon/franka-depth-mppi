"""Actual arrival accounting, independent of solver state and recorded commands."""
import json

class GoalProgress:
    def __init__(self, session, mode):
        self.mode = mode
        self.file = (session / 'arrival-events.jsonl').open('w')
        self.session = session
        self.intervals, self.arrivals = [], []
        self.active = None
        self.cycles = 0
        self.expected = 0
        self.started = False

    def command(self, goal, step, sim_s):
        if self.active is not None:
            if self.active['arrival_sim_s'] is None:
                assert self.mode == 'recorded', 'unreached goal cannot switch'
                self.active.update(end_reason='recorded_schedule', end_sim_s=sim_s)
            self.intervals.append(self.active)
        self.active = dict(goal_index=goal, command_step=step, command_sim_s=sim_s,
                           arrival_step=None, arrival_sim_s=None, duration_s=None,
                           arrival_position_error_m=None, arrival_orientation_error_rad=None,
                           end_reason=None, end_sim_s=None, completed_round_trips=self.cycles)

    def reach(self, step, sim_s, position_error, orientation_error):
        a = self.active
        if a['arrival_sim_s'] is not None:
            return
        assert position_error <= .01 and orientation_error <= .05
        goal = a['goal_index']
        self.arrivals.append(goal)
        # Ignore duplicate arrivals of a goal after a recorded-mode skipped command.
        if goal == self.expected:
            if not self.started:
                self.started = True
            elif goal == 0:
                self.cycles += 1
            self.expected = 1 - goal
        a.update(arrival_step=step, arrival_sim_s=sim_s,
                 duration_s=sim_s-a['command_sim_s'],
                 arrival_position_error_m=position_error,
                 arrival_orientation_error_rad=orientation_error,
                 end_reason='reached', end_sim_s=sim_s,
                 completed_round_trips=self.cycles)
        self.file.write(json.dumps(a)+'\n')
        self.file.flush()

    def finish(self, last):
        if self.active is not None:
            if self.active['arrival_sim_s'] is None:
                self.active.update(end_reason='input_end', end_sim_s=last['sim_s'] if last else 0.,
                                   status='incomplete_at_observation_end',
                                   remaining_position_error_m=last['position_error_m'] if last else None,
                                   remaining_orientation_error_rad=last['orientation_error_rad'] if last else None)
            self.intervals.append(self.active)
        self.file.close()
        (self.session/'goal-intervals.json').write_text(json.dumps(self.intervals,indent=2))
