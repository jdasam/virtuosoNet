import copy

absolute_tempos_keywords = ['adagio', 'grave', 'lento', 'largo', 'larghetto', 'andante', 'andantino', 'moderato',
                            'allegretto', 'allegro', 'vivace', 'accarezzevole', 'languido', 'tempo giusto', 'mesto',
                            'presto', 'prestissimo', 'maestoso', 'lullaby', 'doppio movimento', 'agitato', 'precipitato',
                            'leicht und zart', 'aufgeregt', 'bewegt', 'rasch', 'innig', 'lebhaft', 'geschwind',
                            "d'un rythme souple",
                            'lent', 'large', 'vif', 'animé', 'scherzo', 'menuetto', 'minuetto']
relative_tempos_keywords = ['animato', 'pesante', 'veloce', 'agitato',
                            'acc', 'accel', 'rit', 'ritard', 'ritardando', 'accelerando', 'rall', 'rallentando', 'ritenuto', 'string',
                            'a tempo', 'im tempo', 'stretto', 'slentando', 'meno mosso', 'meno vivo', 'più mosso', 'allargando', 'smorzando', 'appassionato', 'perdendo',
                            'langsamer', 'schneller', 'bewegter',
                             'retenu', 'revenez', 'cédez', 'mesuré', 'élargissant', 'accélerez', 'rapide', 'reprenez  le  mouvement']
relative_long_tempo_keywords = ['meno mosso', 'meno vivo', 'più mosso', 'animato', 'langsamer', 'schneller',
                                'stretto', 'bewegter', 'tranquillo', 'agitato', 'appassionato']
tempo_primo_words = ['tempo i', 'tempo primo', 'erstes tempo', '1er mouvement', '1er mouvt', 'au mouvtdu début', 'au mouvement', 'au mouvt', '1o tempo']

absolute_tempos_keywords += tempo_primo_words
tempos_keywords = absolute_tempos_keywords + relative_tempos_keywords

# tempos_merged_key = ['adagio', 'lento', 'andante', 'andantino', 'moderato', 'allegretto', 'allegro', 'vivace',
#                      'presto', 'prestissimo', 'animato', 'maestoso', 'pesante', 'veloce', 'tempo i', 'lullaby', 'agitato',
#                      ['acc', 'accel', 'accelerando'],['rit', 'ritardando', 'rall', 'rallentando'], 'ritenuto',
#                     'a tempo', 'stretto', 'slentando', 'meno mosso', 'più mosso', 'allargando' ]


absolute_dynamics_keywords = ['pppp','ppp', 'pp', 'p', 'piano', 'mp', 'mf', 'f', 'forte', 'ff', 'fff', 'fp', 'ffp']
relative_dynamics_keywords = ['crescendo', 'diminuendo', 'cresc', 'dim', 'dimin' 'sotto voce',
                              'mezza voce', 'sf', 'fz', 'sfz', 'rf,' 'sffz', 'rf', 'rinf',
                              'con brio', 'con forza', 'con fuoco', 'smorzando', 'appassionato', 'perdendo']
relative_long_dynamics_keywords = ['con brio', 'con forza', 'con fuoco', 'smorzando', 'appassionato', 'perdendo']

dynamics_keywords = absolute_dynamics_keywords + relative_dynamics_keywords
dynamics_merged_keys = ['ppp', 'pp', ['p', 'piano'], 'mp', 'mf', ['f', 'forte'], 'ff', 'fff', 'fp', ['crescendo', 'cresc'],  ['diminuendo', 'dim', 'dimin'],
                        'sotto voce', 'mezza voce', ['sf', 'fz', 'sfz', 'sffz'] ]



class EmbeddingTable():
    def __init__(self):
        self.keywords=[]
        self.embed_key = []

    def append(self, EmbeddingKey):
        self.keywords.append(EmbeddingKey.key)
        self.embed_key.append(EmbeddingKey)

class EmbeddingKey():
    def __init__(self, key_name, vec_idx, value):
        self.key = key_name
        self.vector_index = vec_idx
        self.value = value

class Cresciuto:
    def __init__(self, start, end, type):
        self.xml_position = start
        self.end_xml_position = end
        self.type = type #crescendo or diminuendo
        self.overlapped = 0

def extract_directions_by_keywords(directions, keywords):
    sub_directions =[]

    for dir in directions:
        included = check_direction_by_keywords(dir, keywords)
        if included:
            sub_directions.append(dir)
            # elif dir.type['words'].split('sempre ')[-1] in keywords:
            #     dir.type['dynamic'] = dir.type.pop('words')
            #     dir.type['dynamic'] = dir.type['dynamic'].split('sempre ')[-1]
            #     sub_directions.append(dir)
            # elif dir.type['words'].split('subito ')[-1] in keywords:
            #     dir.type['dynamic'] = dir.type.pop('words')
            #     dir.type['dynamic'] = dir.type['dynamic'].split('subito ')[-1]
            #     sub_directions.append(dir)

    return sub_directions


def check_direction_by_keywords(dir, keywords):
    if dir.type['type'] in keywords:
        return True
    elif dir.type['type'] == 'words' and dir.type['content'] is not None:
        dir_word = word_regularization(dir.type['content'])
        # dir_word = dir.type['content'].replace(',', '').replace('.', '').replace('\n', ' ').replace('(','').replace(')','').lower()
        if dir_word in keywords:
            return True
        else:
            word_split = dir_word.split(' ')
            for w in word_split:
                if w in keywords:
                    # dir.type[keywords[0]] = dir.type.pop('words')
                    # dir.type[keywords[0]] = w
                    return True

        for key in keywords: # words like 'sempre più mosso'
            if len(key) > 2 and key in dir_word:
                return True


def get_dynamics(directions):
    temp_abs_key = absolute_dynamics_keywords
    temp_abs_key.append('dynamic')

    absolute_dynamics = extract_directions_by_keywords(directions, temp_abs_key)
    relative_dynamics = extract_directions_by_keywords(directions, relative_dynamics_keywords)
    abs_dynamic_dummy = []
    for abs in absolute_dynamics:
        if abs.type['content'] == 'fp':
            abs.type['content'] = 'f'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'ffp':
            abs.type['content'] = 'ff'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'sfp':
            abs.type['content'] = 'sf'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)

        if abs.type['content'] in ['sf', 'fz', 'sfz', 'sffz', 'rf', 'rfz']:
            relative_dynamics.append(abs)
        else:
            abs_dynamic_dummy.append(abs)


    absolute_dynamics = abs_dynamic_dummy
    absolute_dynamics, temp_relative = check_relative_word_in_absolute_directions(absolute_dynamics)
    relative_dynamics += temp_relative
    dummy_rel = []
    for rel in relative_dynamics:
        if rel not in absolute_dynamics:
            dummy_rel.append(rel)
    relative_dynamics = dummy_rel

    relative_dynamics.sort(key=lambda x:x.xml_position)
    relative_dynamics = merge_start_end_of_direction(relative_dynamics)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    relative_dynamics_position = [dyn.xml_position for dyn in relative_dynamics]
    cresc_name = ['crescendo', 'diminuendo']
    cresciuto_list = []
    num_relative = len(relative_dynamics)

    for i in range(num_relative):
        rel = relative_dynamics[i]
        if len(absolute_dynamics) > 0:
            index = binaryIndex(absolute_dynamics_position, rel.xml_position)
            rel.previous_dynamic = absolute_dynamics[index].type['content']
            if index + 1 < len(absolute_dynamics):
                rel.next_dynamic = absolute_dynamics[index + 1]  # .type['content']

            else:
                rel.next_dynamic = absolute_dynamics[index]
        if rel.type['type'] == 'dynamic' and not rel.type['content'] in ['rf', 'rfz', 'rffz']: # sf, fz, sfz
            rel.end_xml_position = rel.xml_position + 0.1

        if not hasattr(rel, 'end_xml_position'):
        # if rel.end_xml_position is None:
            for j in range(1, num_relative-i):
                next_rel = relative_dynamics[i+j]
                rel.end_xml_position = next_rel.xml_position
                break

        if len(absolute_dynamics) > 0 and hasattr(rel, 'end_xml_position') and index < len(absolute_dynamics) - 1 and absolute_dynamics[index + 1].xml_position < rel.end_xml_position:
            rel.end_xml_position = absolute_dynamics_position[index + 1]

        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")

        if (rel.type['type'] in cresc_name or crescendo_word_regularization(rel.type['content']) in cresc_name )\
                and (hasattr(rel, 'next_dynamic') and rel.end_xml_position < rel.next_dynamic.xml_position):
            if rel.type['type'] in cresc_name:
                cresc_type = rel.type['type']
            else:
                cresc_type = crescendo_word_regularization(rel.type['content'])
            cresciuto = Cresciuto(rel.end_xml_position, rel.next_dynamic.xml_position, cresc_type)
            cresciuto_list.append(cresciuto)

    return absolute_dynamics, relative_dynamics, cresciuto_list


def get_tempos(directions):
    absolute_tempos = extract_directions_by_keywords(directions, absolute_tempos_keywords)
    relative_tempos = extract_directions_by_keywords(directions, relative_tempos_keywords)
    relative_long_tempos = extract_directions_by_keywords(directions, relative_long_tempo_keywords)

    if (len(absolute_tempos)==0 or absolute_tempos[0].xml_position != 0) \
            and len(relative_long_tempos) > 0 and relative_long_tempos[0].xml_position == 0:
        absolute_tempos.insert(0, relative_long_tempos[0])

    dummy_relative_tempos = []
    for rel in relative_tempos:
        if rel not in absolute_tempos:
            dummy_relative_tempos.append(rel)
    relative_tempos = dummy_relative_tempos

    dummy_relative_tempos = []
    for rel in relative_long_tempos:
        if rel not in absolute_tempos:
            dummy_relative_tempos.append(rel)
    relative_long_tempos = dummy_relative_tempos


    absolute_tempos, temp_relative = check_relative_word_in_absolute_directions(absolute_tempos)
    relative_tempos += temp_relative
    relative_long_tempos += temp_relative
    relative_tempos.sort(key=lambda x:x.xml_position)

    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    num_abs_tempos = len(absolute_tempos)
    num_rel_tempos = len(relative_tempos)

    for abs in absolute_tempos:
        for wrd in tempo_primo_words:
            if wrd in abs.type['content'].lower():
                abs.type['content'] = absolute_tempos[0].type['content']

    for i in range(num_rel_tempos):
        rel = relative_tempos[i]
        if rel not in relative_long_tempos and i+1< num_rel_tempos:
            rel.end_xml_position = relative_tempos[i+1].xml_position
        elif rel in relative_long_tempos:
            for j in range(1, num_rel_tempos-i):
                next_rel = relative_tempos[i+j]
                if next_rel in relative_long_tempos:
                    rel.end_xml_position = next_rel.xml_position
                    break
        if len(absolute_tempos)> 0:
            index = binaryIndex(absolute_tempos_position, rel.xml_position)
            rel.previous_tempo = absolute_tempos[index].type['content']
            if index+1 < num_abs_tempos:
                rel.next_tempo = absolute_tempos[index+1].type['content']
                if not hasattr(rel, 'end_xml_position') or rel.end_xml_position > absolute_tempos_position[index+1]:
                    rel.end_xml_position = absolute_tempos_position[index+1]
        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")

    return absolute_tempos, relative_tempos


def word_regularization(word):
    if word:
        word = word.replace(',', ' ').replace('.', ' ').replace('\n', ' ').replace('(', '').replace(')', '').replace('  ', ' ').lower()
    else:
        word = None
    return word


def crescendo_word_regularization(word):
    word = word_regularization(word)
    if 'decresc' in word:
        word = 'diminuendo'
    elif 'cresc' in word:
        word = 'crescendo'
    elif 'dim' in word:
        word = 'diminuendo'
    return word


def check_relative_word_in_absolute_directions(abs_directions):
    relative_keywords = ['più', 'meno', 'plus', 'moins', 'mehr', 'bewegter', 'langsamer']
    absolute_directions = []
    relative_directions = []
    for dir in abs_directions:
        dir_word = word_regularization(dir.type['content'])
        for rel_key in relative_keywords:
            if rel_key in dir_word:
                relative_directions.append(dir)
                break
        else:
            absolute_directions.append(dir)

    return absolute_directions, relative_directions


def read_all_tempo_vector(path):
    xml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                f == 'musicxml_cleaned.musicxml']
    tempo_embed_table = define_tempo_embedding_table()
    for xmlfile in xml_list:
        print(xmlfile)
        xml_doc = MusicXMLDocument(xmlfile)
        composer_name = copy.copy(path).split('/')[1]
        composer_name_vec = composer_name_to_vec(composer_name)

        xml_notes = extract_notes(xml_doc, melody_only=False, grace_note=True)
        measure_positions = xml_doc.get_measure_positions()
        directions, time_signatures = extract_directions(xml_doc)
        xml_notes = apply_directions_to_notes(xml_notes, directions, time_signatures)
        features = extract_score_features(xml_notes, measure_positions)

    for pair in TEMP_WORDS:
        print(pair)


def keyword_into_onehot(attribute, keywords):
    one_hot = [0] * len(keywords)
    if attribute == None:
        return one_hot
    if attribute in keywords:
        index = find_index_list_of_list(attribute, keywords)
        one_hot[index] = 1


    # for i in range(len(keywords)):
    #     keys = keywords[i]
    #     if type(keys) is list:
    #         for key in keys:
    #             if len(key)>2 and (key.encode('utf-8') in
    word_split = attribute.replace(',', ' ').replace('.', ' ').split(' ')
    for w in word_split:
        index = find_index_list_of_list(w.lower(), keywords)
        if index:
            one_hot[index] = 1

    for key in keywords:

        if isinstance(key, str) and len(key) > 2 and key in attribute:
        # if type(key) is st and len(key) > 2 and key in attribute:
            index = keywords.index(key)
            one_hot[index] = 1

    return one_hot


def direction_words_flatten(note_attribute):
    flatten_words = note_attribute.absolute
    if flatten_words == None: # if absolute direction is None
        flatten_words = ''
    if not note_attribute.relative == []:
        for rel in note_attribute.relative:
            if rel.type['type'] == 'words':
                flatten_words = flatten_words + ' ' + rel.type['content']
            else:
                flatten_words = flatten_words + ' ' + rel.type['type']
    return flatten_words




def dynamic_embedding(dynamic_word, embed_table, len_vec=4):
    dynamic_vector = [0] * len_vec
    # dynamic_vector[0] = 0.5
    keywords = embed_table.keywords
    dynamic_word = word_regularization(dynamic_word)
    # dynamic_word = dynamic_word.replace(',', ' ').replace('.', ' ').replace('\n', ' ').replace('(','').replace(')','').lower()
    applied_words = []

    if dynamic_word == None:
        return dynamic_vector
    if dynamic_word in embed_table.keywords:
        index = find_index_list_of_list(dynamic_word, keywords)
        vec_idx = embed_table.embed_key[index].vector_index
        dynamic_vector[vec_idx] = embed_table.embed_key[index].value

    else:
        # for i in range(len(keywords)):
        #     keys = keywords[i]
        #     if type(keys) is list:
        #         for key in keys:
        #             if len(key)>2 and (key.encode('utf-8') in
        word_split = dynamic_word.split(' ')
        for w in word_split:
            index = find_index_list_of_list(w, keywords)
            if index is not None:
                applied_words.append(w)
                # vec_idx = embed_table.embed_key[index].vector_index
                # dynamic_vector[vec_idx] = embed_table.embed_key[index].value

        for key in keywords:
            if isinstance(key, str) and len(key) > 4 and key in dynamic_word:
                # if type(key) is st and len(key) > 2 and key in attribute:
                applied_words.append(key)
                # index = keywords.index(key)
                # vec_idx = embed_table.embed_key[index].vector_index
                # dynamic_vector[vec_idx] = embed_table.embed_key[index].value

        for word in applied_words:
            for other_word in applied_words:
                if len(word) > 4 and word != other_word and word in other_word:
                    break
            else:
                index = keywords.index(word)
                vec_idx = embed_table.embed_key[index].vector_index
                dynamic_vector[vec_idx] = embed_table.embed_key[index].value

    return dynamic_vector

def define_dynamic_embedding_table():
    embed_table = EmbeddingTable()

    embed_table.append(EmbeddingKey('pppp', 0, -1.1))
    embed_table.append(EmbeddingKey('ppp', 0, -0.9))
    embed_table.append(EmbeddingKey('pp', 0, -0.7))
    embed_table.append(EmbeddingKey('piano', 0, -0.4))
    embed_table.append(EmbeddingKey('p', 0, -0.4))
    embed_table.append(EmbeddingKey('mp', 0, -0.2))
    embed_table.append(EmbeddingKey('mf', 0, 0.2))
    embed_table.append(EmbeddingKey('f', 0, 0.4))
    embed_table.append(EmbeddingKey('forte', 0, 0.4))
    embed_table.append(EmbeddingKey('ff', 0, 0.7))
    embed_table.append(EmbeddingKey('fff', 0, 0.9))

    embed_table.append(EmbeddingKey('più p', 3, -0.5))
    embed_table.append(EmbeddingKey('più piano', 3, -0.5))
    embed_table.append(EmbeddingKey('più pp', 3, -0.7))
    embed_table.append(EmbeddingKey('più f', 3, 0.5))
    embed_table.append(EmbeddingKey('più forte', 3, 0.5))
    embed_table.append(EmbeddingKey('più forte possibile', 0, 1))

    embed_table.append(EmbeddingKey('cresc', 1, 0.7))
    embed_table.append(EmbeddingKey('crescendo', 1, 0.7))
    embed_table.append(EmbeddingKey('allargando', 1, 0.4))
    embed_table.append(EmbeddingKey('dim', 1, -0.7))
    embed_table.append(EmbeddingKey('diminuendo', 1, -0.7))
    embed_table.append(EmbeddingKey('decresc', 1, -0.7))
    embed_table.append(EmbeddingKey('decrescendo', 1, -0.7))

    embed_table.append(EmbeddingKey('smorz', 1, -0.4))

    embed_table.append(EmbeddingKey('poco a poco meno f', 1, -0.2))
    embed_table.append(EmbeddingKey('poco cresc', 1, 0.5))
    embed_table.append(EmbeddingKey('più cresc', 1, 0.85))
    embed_table.append(EmbeddingKey('molto cresc', 1, 1))
    embed_table.append(EmbeddingKey('cresc molto', 1, 1))


    embed_table.append(EmbeddingKey('fz', 2, 0.3))
    embed_table.append(EmbeddingKey('sf', 2, 0.5))
    embed_table.append(EmbeddingKey('sfz', 2, 0.7))
    embed_table.append(EmbeddingKey('ffz', 2, 0.8))
    embed_table.append(EmbeddingKey('sffz', 2, 0.9))

    embed_table.append(EmbeddingKey('rf', 3, 0.3))
    embed_table.append(EmbeddingKey('rinf', 3, 0.3))
    embed_table.append(EmbeddingKey('rinforzando', 3, 0.3))
    embed_table.append(EmbeddingKey('rinforzando molto', 3, 0.7))
    embed_table.append(EmbeddingKey('rinforzando assai', 3, 0.6))
    embed_table.append(EmbeddingKey('rinforz assai', 3, 0.6))
    embed_table.append(EmbeddingKey('molto rin', 3, 0.5))

    embed_table.append(EmbeddingKey('con brio', 3, 0.3))
    embed_table.append(EmbeddingKey('con forza', 3, 0.5))
    embed_table.append(EmbeddingKey('con fuoco', 3, 0.7))
    embed_table.append(EmbeddingKey('con più fuoco possibile', 3, 1))
    embed_table.append(EmbeddingKey('sotto voce', 3, -0.5))
    embed_table.append(EmbeddingKey('mezza voce', 3, -0.3))
    embed_table.append(EmbeddingKey('appassionato', 3, 0.5))
    embed_table.append(EmbeddingKey('più rinforz', 3, 0.5))

    return embed_table


def define_tempo_embedding_table():
    # [abs tempo, abs_tempo_added, tempo increase or decrease, sudden change]
    embed_table = EmbeddingTable()
    embed_table.append(EmbeddingKey('scherzo', 4, 1))
    embed_table.append(EmbeddingKey('menuetto', 4, -1))
    embed_table.append(EmbeddingKey('minuetto', 4, -1))

    #short words
    embed_table.append(EmbeddingKey('rit', 2, -0.5))
    embed_table.append(EmbeddingKey('acc', 2, 0.5))

    embed_table.append(EmbeddingKey('lent', 0, -0.9))
    embed_table.append(EmbeddingKey('lento', 0, -0.9))
    embed_table.append(EmbeddingKey('grave', 0, -0.9))
    embed_table.append(EmbeddingKey('largo', 0, -0.7))
    embed_table.append(EmbeddingKey('languido', 0, -0.6))
    embed_table.append(EmbeddingKey('molto languido', 0, -0.7))
    embed_table.append(EmbeddingKey('adagio', 0, -0.6))
    embed_table.append(EmbeddingKey('larghetto', 0, -0.6))
    embed_table.append(EmbeddingKey('adagietto', 0, -0.55))
    embed_table.append(EmbeddingKey('andante', 0, -0.5))
    embed_table.append(EmbeddingKey('andantino', 0, -0.3))
    embed_table.append(EmbeddingKey('mesto', 0, -0.3))
    embed_table.append(EmbeddingKey('andantino molto', 0, -0.4))
    embed_table.append(EmbeddingKey('maestoso', 0, -0.2))
    embed_table.append(EmbeddingKey('accarezzevole', 0, -0.4))
    embed_table.append(EmbeddingKey('moderato', 0, 0))
    embed_table.append(EmbeddingKey('tempo giusto', 0, 0.1))
    embed_table.append(EmbeddingKey('allegretto', 0, 0.3))
    embed_table.append(EmbeddingKey('allegro', 0, 0.5))
    embed_table.append(EmbeddingKey('allegro assai', 0, 0.6))
    embed_table.append(EmbeddingKey('vivace', 0, 0.6))
    embed_table.append(EmbeddingKey('vivacissimo', 0, 0.8))
    embed_table.append(EmbeddingKey('molto vivace', 0, 0.7))
    embed_table.append(EmbeddingKey('vivace assai', 0, 0.7))
    embed_table.append(EmbeddingKey('presto', 0, 0.8))
    embed_table.append(EmbeddingKey('precipitato', 0, 0.8))
    embed_table.append(EmbeddingKey('prestissimo', 0, 0.9))

    embed_table.append(EmbeddingKey('doppio movimento', 0, 0.6))
    embed_table.append(EmbeddingKey('molto allegro', 0, 0.6))
    embed_table.append(EmbeddingKey('allegro molto', 0, 0.6))
    embed_table.append(EmbeddingKey('allegro ma non troppo', 0, 0.4))
    embed_table.append(EmbeddingKey('più presto possibile', 0, 1))
    embed_table.append(EmbeddingKey('largo e mesto', 0, -0.8))
    embed_table.append(EmbeddingKey('non troppo presto', 0, 0.75))
    embed_table.append(EmbeddingKey('andante con moto', 0, -0.4))
    embed_table.append(EmbeddingKey('allegretto vivace', 0, 0.4))
    embed_table.append(EmbeddingKey('adagio molto', 0, -0.9))
    embed_table.append(EmbeddingKey('adagio ma non troppo', 0, -0.5))


    embed_table.append(EmbeddingKey('a tempo', 1, 0))
    embed_table.append(EmbeddingKey('im tempo', 1, 0))
    embed_table.append(EmbeddingKey('meno mosso', 1, -0.6))
    embed_table.append(EmbeddingKey('meno vivo', 1, -0.6))
    embed_table.append(EmbeddingKey('più lento', 1, -0.5))

    embed_table.append(EmbeddingKey('animato', 1, 0.5))
    embed_table.append(EmbeddingKey('più animato', 1, 0.6))
    embed_table.append(EmbeddingKey('molto animato', 1, 0.8))
    embed_table.append(EmbeddingKey('agitato', 1, 0.4))
    embed_table.append(EmbeddingKey('più mosso', 1, 0.6))
    embed_table.append(EmbeddingKey('stretto', 1, 0.5))
    embed_table.append(EmbeddingKey('appassionato', 1, 0.2))
    embed_table.append(EmbeddingKey('più moderato', 1, -0.1))
    embed_table.append(EmbeddingKey('più allegro', 1, 0.8))
    embed_table.append(EmbeddingKey('più allegro quasi presto', 1, 1))
    embed_table.append(EmbeddingKey('più largo', 1, -0.5))

    embed_table.append(EmbeddingKey('poco meno mosso', 1, -0.3))
    embed_table.append(EmbeddingKey('poco più mosso', 1, 0.3))
    embed_table.append(EmbeddingKey('poco più vivace', 1, 0.3))
    embed_table.append(EmbeddingKey('molto agitato', 1, 0.8))
    embed_table.append(EmbeddingKey('tranquillo', 1, -0.2))
    embed_table.append(EmbeddingKey('meno adagio', 1, 0.2))
    embed_table.append(EmbeddingKey('più adagio', 1, -0.2))


    embed_table.append(EmbeddingKey('riten', 3, -0.5))
    embed_table.append(EmbeddingKey('ritenuto', 3, -0.5))
    embed_table.append(EmbeddingKey('più riten', 3, -0.7))
    embed_table.append(EmbeddingKey('poco riten', 3, -0.3))
    embed_table.append(EmbeddingKey('poco ritenuto', 3, -0.3))

    # French
    embed_table.append(EmbeddingKey('très grave', 0, -1))
    embed_table.append(EmbeddingKey('très lent', 0, -1))
    # embed_table.append(EmbeddingKey('marche funèbre', 0, -0.8))
    embed_table.append(EmbeddingKey('large', 0, -0.7))
    embed_table.append(EmbeddingKey("Assez doux, mais d'une sonoritè large", 0, -0.6))
    embed_table.append(EmbeddingKey('assez vif', 0, 0.6))
    embed_table.append(EmbeddingKey('assez animé', 0, 0.7))
    embed_table.append(EmbeddingKey("d'un rythme souple", 0, 0))

    embed_table.append(EmbeddingKey('un peau retenu', 1, -0.3))
    embed_table.append(EmbeddingKey('retenez', 1, -0.5))
    embed_table.append(EmbeddingKey('en élargissant beaucoup', 2, -0.4))
    embed_table.append(EmbeddingKey('plus lent', 1, -0.5))
    embed_table.append(EmbeddingKey('un peu plus lent', 1, -0.3))
    embed_table.append(EmbeddingKey('encore plus lent', 1, -0.5))
    embed_table.append(EmbeddingKey('cédez', 1, -0.4))
    embed_table.append(EmbeddingKey('un peu retenu', 1, -0.2))
    embed_table.append(EmbeddingKey('Un pe moins vif', 1, -0.2))
    embed_table.append(EmbeddingKey('cédez légèrement', 1, -0.2))
    embed_table.append(EmbeddingKey('mesuré', 1, 0))
    embed_table.append(EmbeddingKey('au mouvt', 1, 0))
    embed_table.append(EmbeddingKey('reprenez le mouvement', 1, 0))
    embed_table.append(EmbeddingKey('Un peu plus vif', 1, 0.2))
    embed_table.append(EmbeddingKey('en accélérant', 1, 0.3))
    embed_table.append(EmbeddingKey('en animant', 1, 0.6))
    embed_table.append(EmbeddingKey('rapide', 1, 0.6))
    embed_table.append(EmbeddingKey('rapido molto', 1, 0.8))

    embed_table.append(EmbeddingKey('accélerez', 2, 0.5))
    embed_table.append(EmbeddingKey('sans ralentir', 2, 0.01))

    # German
    embed_table.append(EmbeddingKey('sehr langsam', 0, -0.55))
    embed_table.append(EmbeddingKey('sehr innig', 0, -0.5))
    embed_table.append(EmbeddingKey('innig', 0, -0.4))
    embed_table.append(EmbeddingKey('leicht und zart', 0, 0.2))
    embed_table.append(EmbeddingKey('sehr lebhaft', 0, 0.6))
    embed_table.append(EmbeddingKey('sehr aufgeregt', 0, 0.6))
    embed_table.append(EmbeddingKey('sehr rasch', 0, 0.8))
    embed_table.append(EmbeddingKey('Sehr innig und nicht zu rasch', 0, -0.8))
    embed_table.append(EmbeddingKey('lebhaftig', 0, 0.5))
    embed_table.append(EmbeddingKey('nicht zu geschwind', 0, -0.3))

    embed_table.append(EmbeddingKey('langsamer', 1, -0.5))
    embed_table.append(EmbeddingKey('etwas langsamer', 1, -0.3))
    embed_table.append(EmbeddingKey('bewegter', 1, 0.4))
    embed_table.append(EmbeddingKey('schneller', 1, 0.5))

    embed_table.append(EmbeddingKey('allargando', 2, -0.2))
    embed_table.append(EmbeddingKey('ritardando', 2, -0.5))
    embed_table.append(EmbeddingKey('rit', 2, -0.5))
    embed_table.append(EmbeddingKey('ritar', 2, -0.5))
    embed_table.append(EmbeddingKey('rallentando', 2, -0.5))
    embed_table.append(EmbeddingKey('rall', 2, -0.5))
    embed_table.append(EmbeddingKey('slentando', 2, -0.3))
    embed_table.append(EmbeddingKey('accel', 2, 0.5))
    embed_table.append(EmbeddingKey('accelerando', 2, 0.5))
    embed_table.append(EmbeddingKey('smorz', 2, -0.5))
    embed_table.append(EmbeddingKey('string', 2, 0.4))
    embed_table.append(EmbeddingKey('stringendo', 2, 0.4))
    embed_table.append(EmbeddingKey('molto stringendo', 2, 0.7))
    embed_table.append(EmbeddingKey('stringendo molto', 2, 0.7))


    embed_table.append(EmbeddingKey('poco rall', 2, -0.3))
    embed_table.append(EmbeddingKey('poco rit', 2, -0.3))

    # non troppo presto
    # energico
    # tempo di marcia
    # marcia funbre
    # ben marcato
    # rinforzando assai
    # rinforzando molto
    # piu presto possibile
    # mesto
    # tempo giusto
    # sempre più largo
    # plus lent
    # 1er Mouvement

    return embed_table


def find_index_list_of_list(element, in_list):
    # isuni = isinstance(element, unicode) # for python 2.7
    if element in in_list:
        return in_list.index(element)
    else:
        for li in in_list:
            if isinstance(li, list):
                # if isuni:
                #     li = [x.decode('utf-8') for x in li]
                if element in li:
                    return in_list.index(li)

    return None


def merge_start_end_of_direction(directions):
    for i in range(len(directions)):
        dir = directions[i]
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] == "stop":
            for j in range(i):
                prev_dir = directions[i-j-1]
                prev_type_name = prev_dir.type['type']
                if type_name == prev_type_name and prev_dir.type['content'] == "start" and dir.staff == prev_dir.staff:
                    prev_dir.end_xml_position = dir.xml_position
                    break
    dir_dummy = []
    for dir in directions:
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] != "stop":
            # directions.remove(dir)
            dir_dummy.append(dir)
        elif type_name == 'words':
            dir_dummy.append(dir)
    directions = dir_dummy
    return directions


def binaryIndex(alist, item):
    first = 0
    last = len(alist)-1
    midpoint = 0

    if(item< alist[first]):
        return 0

    while first<last:
        midpoint = (first + last)//2
        currentElement = alist[midpoint]

        if currentElement < item:
            if alist[midpoint+1] > item:
                return midpoint
            else: first = midpoint +1;
            if first == last and alist[last] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint -1
        else:
            if midpoint +1 ==len(alist):
                return midpoint
            while alist[midpoint+1] == item:
                midpoint += 1
                if midpoint + 1 == len(alist):
                    return midpoint
            return midpoint
    return last