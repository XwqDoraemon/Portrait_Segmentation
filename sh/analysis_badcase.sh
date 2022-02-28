dataset="/hy-tmp/dataset"
set_name="testing"
checkpoint="/hy-tmp/output_seg_bkg/checkpoint_best.pkl"
save="False"
python bins/analysis_badcase.py -c $checkpoint \
                 -d $dataset \
                 -s $set_name \
                 --save $save