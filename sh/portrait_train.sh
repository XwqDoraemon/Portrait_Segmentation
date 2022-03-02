data_root_dir="/hy-tmp/dataset"
output_dir="/hy-tmp/output_seg"
ext_dir="/hy-tmp/34427dataset"
# fusion_dir="/home/myjian/Workespace/OutputDir/portrait_train/fusion_dir"
python src/portrait_train.py --data_root_dir $data_root_dir \
                --output_dir $output_dir \
                --ext_dir $ext_dir \
                # --fusion_dir $fusion_dir