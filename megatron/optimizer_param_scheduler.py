# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""

import math

from megatron import print_rank_0

class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay"""


    def __init__(self, optimizer, init_lr, max_lr, min_lr, constant_lr,
                 lr_warmup_steps, lr_decay_steps, num_repeats, end_steps, 
                 constant_steps, cooldown_steps, sqrt_scale, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        self.constant_lr = constant_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.num_repeats = num_repeats
        self.constant_steps = constant_steps
        self.cooldown_steps = cooldown_steps
        self.sqrt_scale = sqrt_scale
        self.end_steps = end_steps
        assert self.lr_decay_steps > 0
        assert self.num_repeats > 0
        assert self.lr_warmup_steps < self.lr_decay_steps
        if lr_decay_style == 'invsqrt-inf':
            assert self.end_steps > 0
            assert self.cooldown_steps > 0
            assert self.sqrt_scale > 0
            assert self.constant_steps >= 0
        

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == 'constant':
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""
        
        if self.lr_decay_style != 'invsqrt-inf':
            self.num_repeats = 1

        warmup = self.lr_warmup_steps / self.end_steps
        
        lr_warmup_steps = warmup * int(self.end_steps / self.num_repeats)

        repeat_step_interval = int(self.end_steps / self.num_repeats)


        # Use linear warmup for the initial part.
        if lr_warmup_steps > 0 and self.num_steps <= lr_warmup_steps:
            return (
                self.init_lr
                + (
                    (self.max_lr - self.init_lr)
                    * float(self.num_steps)
                    / float(lr_warmup_steps)
                )
            )
        
        elif self.lr_decay_style == "invsqrt-inf" \
            and lr_warmup_steps > 0 and (self.num_steps % repeat_step_interval) <= lr_warmup_steps:
            return (
                self.min_lr
                + (
                    (self.max_lr - self.min_lr)
                    * float(self.num_steps % repeat_step_interval)
                    / float(lr_warmup_steps)
                )
            )
           
        else:
            # If decay stile is inverse sqrt infinite:
            if self.lr_decay_style == 'invsqrt-inf':    
                # stuff

                num_steps = (self.num_steps % repeat_step_interval) - lr_warmup_steps

                end_steps_ = repeat_step_interval - lr_warmup_steps
                constant_steps_ = ((self.constant_steps + self.cooldown_steps) / self.end_steps) * int(self.end_steps / self.num_repeats)
                cooldown_steps_ = (self.cooldown_steps / self.end_steps) * int(self.end_steps / self.num_repeats)
                # sqrt_scale_ = (self.sqrt_scale / self.end_steps) * int(self.end_steps / self.num_repeats)


                if num_steps <= constant_steps_:
                    if num_steps <= cooldown_steps_:

                        def inv_sqrt(x):
                            return self.max_lr/math.sqrt((x + 1)/1)

                        def y_shifted(x, func, A, B_new, x_start, x_end):
                            return ((B_new - A) / (func(x_end) - func(x_start))) * func(x) + A - ((B_new - A) / (func(x_end) - func(x_start))) * func(x_start)

                        def x_shifted(x, func, x_start, x_end, x_end_new):
                            k = (x_end_new - x_start) / (x_end - x_start)
                            return func(((x - x_start) / k) + x_start)

                        y_shifted_func = lambda x: y_shifted(x, inv_sqrt, self.max_lr, self.constant_lr, 0, self.sqrt_scale)
                        x_shifted_func = lambda x: x_shifted(x, y_shifted_func, 0, self.sqrt_scale, cooldown_steps_)
                        lr = x_shifted_func(num_steps)
                        return lr
                    else:
                    # Stay at constant LR
                        lr = self.constant_lr
                    return lr
                else:
                    # Go from constant iters to min LR in remaining iters
                    end_steps__ = end_steps_ - constant_steps_
                    num_steps = num_steps - constant_steps_
                    exp_factor = -math.log(self.min_lr/self.constant_lr) / end_steps__
                    lr = self.constant_lr * math.exp(-1* exp_factor * num_steps)
                    return lr


            # All other decay styles:
            else:  

                # If the learning rate is constant, just return the initial value.
                if self.lr_decay_style == 'constant':
                    return self.max_lr
                
                # For any steps larger than `self.lr_decay_steps`, use `self.min_lr`.
                if self.num_steps > self.lr_decay_steps:
                        return self.min_lr
                    
                # If we are done with the warmup period, use the decay style.
                if self.lr_decay_style == 'inverse-square-root':
                    warmup_steps = max(self.lr_warmup_steps, 1)
                    num_steps = max(self.num_steps, 1)
                    lr = self.max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
                    return max(self.min_lr, lr)

                num_steps_ = self.num_steps - self.lr_warmup_steps
                decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
                decay_ratio = float(num_steps_) / float(decay_steps_)
                assert decay_ratio >= 0.0
                assert decay_ratio <= 1.0
                delta_lr = self.max_lr - self.min_lr

                if self.lr_decay_style == 'linear':
                    coeff = (1.0 - decay_ratio)
                elif self.lr_decay_style == 'cosine':
                    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
                else:       
                    raise Exception('{} decay style is not supported.'.format(
                        self.lr_decay_style))

                return self.min_lr + coeff * delta_lr


    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_lr = self.get_lr()
        new_wd = self.get_wd()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr * group.get('lr_mult', 1.0)
            group['weight_decay'] = new_wd * group.get('wd_mult', 1.0)


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps
        }

        if self.lr_decay_style == 'invsqrt-inf':
            state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps,
            'constant_lr': self.constant_lr,
            'num_repeats': self.num_repeats,
            'constant_steps': self.constant_steps,
            'cooldown_steps': self.cooldown_steps,
            'sqrt_scale': self.sqrt_scale,
            'end_steps': self.end_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, \
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')
        
        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            lr_warmup_steps_ = sd['warmup_iter']
        elif 'warmup_steps' in sd:
            lr_warmup_steps_ = sd['warmup_steps']
        else:
            lr_warmup_steps_ = sd['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps,
                                                lr_warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            lr_decay_steps_ = sd['end_iter']
        elif 'decay_steps' in sd:
            lr_decay_steps_  = sd['decay_steps']
        else:
            lr_decay_steps_ = sd['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, lr_decay_steps_,
                                               'total number of iterations')

        if 'decay_style' in sd:
            lr_decay_style_ = sd['decay_style']
        else:
            lr_decay_style_ = sd['lr_decay_style']
            
        self.lr_decay_style = self._check_and_set(self.lr_decay_style,
                                               lr_decay_style_,
                                               'learning rate decay style')
        if lr_decay_style_ == 'invsqrt-inf':

                self.constant_lr = self._check_and_set(self.constant_lr,
                                               sd['constant_lr'],
                                               'value of learning rate in constant phase')
                self.num_repeats = self._check_and_set(self.num_repeats,
                                               sd['num_repeats'],
                                               'number of cycles')
                self.constant_steps = self._check_and_set(self.constant_steps,
                                               sd['constant_steps'],
                                               'number of steps of constant learning rate phase')
                self.cooldown_steps = self._check_and_set(self.cooldown_steps,
                                               sd['cooldown_steps'],
                                               'number of steps of inverse sqrt cooldown phase')
                self.sqrt_scale = self._check_and_set(self.sqrt_scale,
                                               sd['sqrt_scale'],
                                               'rate of decay in inverse sqrt phase')
                self.end_steps = self._check_and_set(self.end_steps,
                                               sd['end_steps'],
                                               'total number of steps')


        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        self.step(increment=num_steps)


        if 'start_wd' in sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                sd['start_wd'],
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                                sd['end_wd'],
                                                "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                sd['wd_incr_steps'],
                                                "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                sd['wd_incr_style'],
                                                "weight decay incr style")
            







