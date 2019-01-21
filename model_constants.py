SCORE_INPUT = 77 #score information only
DROP_OUT = 0.5
TOTAL_OUTPUT = 16

NUM_PRIME_PARAM = 11
NUM_TEMPO_PARAM = 1
VEL_PARAM_IDX = 1
DEV_PARAM_IDX = 2
PEDAL_PARAM_IDX = 3
num_second_param = 0
num_trill_param = 5
num_voice_feed_param = 0 # velocity, onset deviation
num_tempo_info = 0
num_dynamic_info = 0 # distance from marking, dynamics vector 4, mean_piano, forte marking and velocity = 4
is_trill_index_score = -11
is_trill_index_concated = -11 - (NUM_PRIME_PARAM + num_second_param)


# test_piece_list = [('schumann', 'Schumann'),
#                 ('mozart545-1', 'Mozart'),
#                 ('chopin_nocturne', 'Chopin'),
#                 ('chopin_fantasie_impromptu', 'Chopin'),
#                 ('cho_waltz_69_2', 'Chopin'),
#                 ('lacampanella', 'Liszt'),
#                 ('bohemian_rhapsody', 'Liszt')
#                 ]

test_piece_list = [('chopin_etude_10_2', 'Chopin'),
                ('chopin_barcarolle', 'Chopin'),
                ('chopin_scherzo', 'Chopin'),
                ('haydn_keyboard_31_1', 'Haydn'),
                ('haydn_keyboard_49_1', 'Haydn'),
                ('bps_5_1', 'Beethoven'),
                ('bps_7_2', 'Beethoven'),
                ('bps_27_1', 'Beethoven'),
                ('bps_31_2', 'Beethoven'),
                ('schubert_impromptu', 'Schubert'),
                ('schubert_ps', 'Schubert'),
                ('liszt_5', 'Liszt'),
                ('liszt_9', 'Liszt'),
                ('liszt_pag', 'Liszt'),
                ('chopin_prelude_1', 'Chopin'),
                ('chopin_prelude_4', 'Chopin'),
                ('chopin_prelude_5', 'Chopin'),
                ('chopin_prelude_6', 'Chopin'),
                ('chopin_prelude_8', 'Chopin')]

emotion_folder_path = 'test_pieces/emotionNet/'
emotion_key_list = ['OR', 'Anger', 'Enjoy', 'Relax', 'Sad']
emotion_data_path  = [('Bach_Prelude_1', 'Bach', 1),
                      ('Clementi_op.36-1_mov3', 'Haydn', 3),
                      ('Kuhlau_op.20-1_mov1', 'Haydn', 2),
                      ]