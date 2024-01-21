from virtuoso.inference import InferenceModel

from virtuoso.parser import get_parser
from virtuoso.pyScoreParser.data_class import ScoreData, PerformData, PieceMeta, PieceData
from pathlib import Path
from virtuoso.utils import handle_args

parser = get_parser()
args = parser.parse_args(
    args=["--checkpoint=../virtuosonet_checkpoints/yml_path=ymls/han_measnote_gru.yml meas_note=True delta_weight=5.0 delta_loss=True vel_balance_loss=True intermediate_loss=False_220203-122932/checkpoint_last.pt",
          "--xml_path=test_pieces/_ConcertCustomization/kreisleriana_3/KREISLERIANA_No.3_.xml",
          "--composer=Schumann",
          "--device=cpu"]
)


# args, _, _ = handle_args(args)

inferencer = InferenceModel(args.checkpoint, args.device, args.output_path, {'multi_instruments': True, 'bool_pedal': False})

for i in range(7,8):
  i = str(i)
  xml_path = Path(f'test_pieces/_ConcertCustomization/Kreisleriana-ver3/KREISLERIANA_No.{i}.musicxml')
  perf_midi_path = Path(f'test_pieces/_ConcertCustomization/Kreisleriana-ver3/{i}-b.mid')
  save_path = perf_midi_path.parent / f'four_voice_{i}.mid'
  score, x, y, edges, note_location = inferencer.get_score_perf_pair_data(xml_path, perf_midi_path, args.composer, save_data=True)
  outputs = inferencer.scale_model_prediction_to_original(y)
  output_features = inferencer.model_prediction_to_feature(outputs)
  inferencer.midi_decoder(score, note_location, output_features, save_path)



# inferencer.infer_xml(args.xml_path, args.composer, args.qpm_primo)
# midi_path = Path(args.xml_path).parent / 'midi_cleaned.mid'
# perf_midi_path = ['test_pieces/_ConcertCustomization/kreisleriana_3/3-a.mid', 'test_pieces/_ConcertCustomization/kreisleriana_3/3-f.mid']



# inferencer.interpolate_performance(args.xml_path, perf_midi_path, args.composer, 'test_result/schumann_immersive_interpol.mid')