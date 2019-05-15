SCORE_INPUT = 78 #score information only
DROP_OUT = 0.2
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


MEAS_TEMPO_IDX = 13
BEAT_TEMPO_IDX = 11


# test_piece_list = [('schumann', 'Schumann'),
#                 ('mozart545-1', 'Mozart'),
#                 ('chopin_nocturne', 'Chopin'),
#                 ('chopin_fantasie_impromptu', 'Chopin'),
#                 ('cho_waltz_69_2', 'Chopin'),
#                 ('lacampanella', 'Liszt'),
#                 ('bohemian_rhapsody', 'Liszt')
#                 ]



VALID_LIST =['Bach/Prelude/bwv_865/',
             'Bach/Prelude/bwv_874/',
             'Bach/Fugue/bwv_865/',
             'Bach/Fugue/bwv_874/',
             'Chopin/Etudes_op_10/10/',
             'Chopin/Etudes_op_25/4/',
             'Chopin/Scherzos/31/',
             'Chopin/Sonata_3/4th/',
             'Mozart/Piano_Sonatas/12-3/',
             'Haydn/Keyboard_Sonatas/46-1/',
             'Rachmaninoff/Preludes_op_23/4/',
             'Beethoven/Piano_Sonatas/3-1/',
             'Beethoven/Piano_Sonatas/12-1/',
             'Beethoven/Piano_Sonatas/15-4/',
             'Beethoven/Piano_Sonatas/21-2/',
             'Beethoven/Piano_Sonatas/30-1/',
             'Schumann/Kreisleriana/5/',
             'Schubert/Impromptu_op.90_D.899/2/',
             'Liszt/Annees_de_pelerinage_2/1_Gondoliera',
             'Liszt/Transcendental_Etudes/4',
             'Liszt/Concert_Etude_S145/2',
             ]

TEST_LIST = ['Bach/Prelude/bwv_858/',
             'Bach/Prelude/bwv_891/',
             'Bach/Fugue/bwv_858/',
             'Bach/Fugue/bwv_891/',
             'Chopin/Etudes_op_10/2/',
             'Chopin/Etudes_op_10/12/',
             'Chopin/Etudes_op_25/12/',
             # 'Chopin/Barcarolle/',
             'Chopin/Scherzos/39/',
             'Haydn/Keyboard_Sonatas/31-1/',
             'Haydn/Keyboard_Sonatas/49-1/',
             'Beethoven/Piano_Sonatas/5-1/',
             'Beethoven/Piano_Sonatas/7-2/',
             'Beethoven/Piano_Sonatas/17-1/',
             # 'Beethoven/Piano_Sonatas/17-1_no_repeat/',
             'Beethoven/Piano_Sonatas/27-1/',
             'Beethoven/Piano_Sonatas/31-2/',
             'Schubert/Impromptu_op.90_D.899/3/',
             'Schubert/Piano_Sonatas/664-1/',
             'Liszt/Transcendental_Etudes/5/',
             'Liszt/Transcendental_Etudes/9/',
             'Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations/'
             ]


test_piece_list = [
                ('bps_5_1', 'Beethoven'),
                ('bps_27_1', 'Beethoven'),
                ('bps_17_1', 'Beethoven'),
                ('bps_7_2', 'Beethoven'),
                ('bps_31_2', 'Beethoven'),
                ('bwv_858_prelude', 'Bach'),
                ('bwv_891_prelude', 'Bach'),
                ('bwv_858_fugue', 'Bach'),
                ('bwv_891_fugue', 'Bach'),
                ('schubert_ps', 'Schubert'),
                ('haydn_keyboard_31_1', 'Haydn'),
                ('haydn_keyboard_49_1', 'Haydn'),
                # ('schubert_impromptu', 'Schubert'),
                # ('mozart545-1', 'Mozart'),
                # ('mozart_symphony', 'Mozart'),
                ('liszt_pag', 'Liszt'),
                ('liszt_5', 'Liszt'),
                ('liszt_9', 'Liszt'),
                ('chopin_etude_10_2', 'Chopin'),
                ('chopin_etude_10_12', 'Chopin'),
                ('chopin_etude_25_12', 'Chopin'),
                # ('cho_waltz_69_2', 'Chopin'),
                # ('chopin_nocturne', 'Chopin'),
                # ('cho_noc_9_1', 'Chopin'),
                # ('chopin_prelude_1', 'Chopin'),
                # ('chopin_prelude_4', 'Chopin'),
                # ('chopin_prelude_5', 'Chopin'),
                # ('chopin_prelude_6', 'Chopin'),
                # ('chopin_prelude_8', 'Chopin'),
                # ('chopin_prelude_15', 'Chopin'),
                # ('kiss_the_rain', 'Chopin'),
                # ('bohemian_rhapsody', 'Liszt'),
                # ('chopin_fantasie_impromptu', 'Chopin'),
                # ('schumann', 'Schumann'),
                ('chopin_barcarolle', 'Chopin'),
                # ('chopin_scherzo', 'Chopin'),
                   ]

emotion_folder_path = 'test_pieces/emotionNet/'
emotion_key_list = ['OR', 'Anger', 'Enjoy', 'Relax', 'Sad']
emotion_data_path  = [('Bach_Prelude_1', 'Bach', 1),
                      ('Clementi_op.36-1_mov3', 'Haydn', 3),
                      ('Kuhlau_op.20-1_mov1', 'Haydn', 2),
                      ]