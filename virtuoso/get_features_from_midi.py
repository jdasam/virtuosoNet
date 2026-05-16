import argparse
from .pyScoreParser.data_class import PieceData
from .pyScoreParser.feature_extraction import PerformExtractor
from .pyScoreParser.data_for_training import VNET_OUTPUT_KEYS

def load_xml_and_perf_midi(xml_path, midi_path, composer):
    piece = PieceData(xml_path, [midi_path], composer=composer)
    perf_extractor = PerformExtractor(VNET_OUTPUT_KEYS)
    perf_features = perf_extractor.extract_perform_features(piece.performances[0])

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str,
                        default='./test_pieces/beethoven5_4hands.musicxml', help="xml_path")
    parser.add_argument("--midi_path", type=str,
                        default="./test_result/b5_final.mid/", help="perf midi path")
    parser.add_argument("--composer", type=str,
                        default="Beethoven")

    args = parser.parse_args()
    load_xml_and_perf_midi(args.xml_path, args.midi_path, args.composer)
