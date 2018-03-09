#ADD args.outdir to specify where to store the pickled dataframes"

PERCENT_KEEP=0
VOCAB_THRESH=5

echo "MPE"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/mpe/cl_mpe_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/mpe/cl_mpe_val_lbl_file --pred_lbl /export/a13/ahaldar/nli-hypothes-only/output/mpe/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src MPE

echo "SNLI"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/snli_1.0/cl_snli_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/snli_1.0/cl_snli_val_lbl_file  --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/snli_1.0/max-pool-sgd/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src SNLI

echo "MNLI-matched"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/multinli_1.0/cl_multinli_dev_matched_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/multinli_1.0/cl_multinli_dev_matched_lbl_file --pred_lbl  /export/b02/apoliak/nli-hypothes-only/output/mnli/matched/max-pool-sgd/batchsize-64/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src MNLI-matched

echo "MNLI-mismatched"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/multinli_1.0/cl_multinli_dev_mismatched_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/multinli_1.0/cl_multinli_dev_mismatched_lbl_file --pred_lbl  /export/b02/apoliak/nli-hypothes-only/output/mnli/mismatched/max-pool-sgd/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src MNLI-mismatched

echo "ADD-1"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/add-one-rte/cl_add_one_rte_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/add-one-rte/cl_add_one_rte_val_lbl_file --pred_lbl /export/a13/ahaldar/nli-hypothes-only/output/add-one/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src ADD-1


echo "Scitail"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/scitail/cl_scitail_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/scitail/cl_scitail_val_lbl_file --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/scitail/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src SciTail

echo "JOCI"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/joci/cl_joci_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/joci/cl_joci_val_lbl_file --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/joci/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src JOCI

echo "SPR"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_sprl_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_sprl_val_lbl_file --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/sprl/max-pool-sgd/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src SPR

echo "FN+"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_fnplus_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_fnplus_val_lbl_file --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/fnplus/max-pool-sgd/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src FN+

echo "DPR"
python plot-vocab.py --hyp_src /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_dpr_val_source_file --gold_lbl /export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_dpr_val_lbl_file --pred_lbl /export/b02/apoliak/nli-hypothes-only/output/dpr/max-pool-sgd/val_preds --preds 0 --data_split dev --vocab_thresh $VOCAB_THRESH --percent_keep $PERCENT_KEEP --data_src DPR

