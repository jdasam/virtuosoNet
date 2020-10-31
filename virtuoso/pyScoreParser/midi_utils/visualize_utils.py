import numpy as np
import matplotlib.pyplot as plt


def my_imshow(array, interpolation='nearest', origin='bottom', aspect='auto', cmap='gray', **kwargs):
  plt.imshow(array, interpolation=interpolation, origin=origin, aspect=aspect, cmap=cmap, **kwargs)


note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def midi_name(midi_num):
  octave = str(midi_num // 12 - 1)
  note = note_names[(midi_num + 3) % 12]
  return octave, note


def draw_piano_roll(roll, draw_range=None, fps=10, midi_min=21, midi_max=108):
  if not draw_range:
    draw_range = [roll.shape[0], roll.shape[1]]
  plt.imshow(roll[draw_range[0]: draw_range[1], :].T, interpolation='nearest', aspect='auto',
             origin='lower', cmap=plt.get_cmap('gray_r'))
  tick_range = range((draw_range[1] - draw_range[0]) // fps)

  # draw guide lines (octave line)
  number = 12
  cmap = plt.get_cmap('Paired')
  colors = [cmap(i) for i in np.linspace(0, 1, number)]

  n_midi = midi_max - midi_min + 1
  edge = range(12 - midi_min % 12, n_midi, 12)
  for el in edge:
    plt.plot(draw_range, [el, el], color='red', linewidth=1, linestyle="--", alpha=0.8)
  midi_ticks = [midi_name(el)[1] + midi_name(el)[0] for el in range(midi_min, midi_max)]
  '''
  # only shows white notes names
  for n in range(len(midi_ticks)):
    if '#' in midi_ticks[n]:
      midi_ticks[n] = ''
  '''

  for n in range(n_midi):
    plt.plot(draw_range, [n, n], color=colors[n % 12], linewidth=7, linestyle="-", alpha=0.07)
  plt.yticks(range(n_midi), midi_ticks)
  plt.ylim([0 - 0.5, n_midi - 0.5])

  plt.xticks([el * fps for el in tick_range], [el + draw_range[0] for el in tick_range])
  plt.xlabel('time(sec)')
  plt.xlim(draw_range)