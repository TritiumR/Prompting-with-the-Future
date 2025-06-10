NAME=$1
INSTRUCTION=$2
SUFFIX=world

cd utils/reconstruct
python video_to_png.py --name $NAME
DATASET_PATH=./colmap/$NAME

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

cd ../../gaussians/

colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/extracted_frames \
    --quality extreme \
    --single_camera 1 \
    --use_gpu 1

colmap image_undistorter \
  --image_path $DATASET_PATH/extracted_frames \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path ./data/colmap/$NAME/ \
  --output_type COLMAP


mkdir ./data/colmap/$NAME/sparse/0
mv ./data/colmap/$NAME/sparse/images.bin ./data/colmap/$NAME/sparse/0/images.bin
mv ./data/colmap/$NAME/sparse/cameras.bin ./data/colmap/$NAME/sparse/0/cameras.bin
mv ./data/colmap/$NAME/sparse/points3D.bin ./data/colmap/$NAME/sparse/0/points3D.bin

cd ../utils/reconstruct
python chess_board_pose.py $NAME $SUFFIX

cd ../../gaussians/

mkdir ./data/colmap/"$NAME"_"$SUFFIX"/sparse
mkdir ./data/colmap/"$NAME"_"$SUFFIX"/sparse/0
cp -r ./data/colmap/$NAME/images ./data/colmap/"$NAME"_"$SUFFIX"/images
cp ./data/colmap/$NAME/sparse/0/cameras.bin ./data/colmap/"$NAME"_"$SUFFIX"/sparse/0/cameras.bin

colmap model_aligner \
    --input_path data/colmap/$NAME/sparse/0 \
    --output_path data/colmap/"$NAME"_"$SUFFIX"/sparse/0 \
    --ref_images_path data/colmap/"$NAME"_"$SUFFIX"/ref.txt \
    --ref_is_gps 0 \
    --alignment_type ecef \
    --robust_alignment_max_error 3.0

python train.py -s data/colmap/"$NAME"_"$SUFFIX" --model_path ./output/"$NAME"_"$SUFFIX" --ip 127.0.0.1
python render.py -s data/colmap/"$NAME"_"$SUFFIX" -m ./output/"$NAME"_"$SUFFIX" --render_path --skip_test --skip_train

cd ../utils/reconstruct

python render_video.py --name "$NAME"_"$SUFFIX"

cd ../../sam2

conda run --no-capture-output -n sam2 python segment.py --name "$NAME"_"$SUFFIX" --instruction "$INSTRUCTION"

cd ../

python decompose.py --name "$NAME"_"$SUFFIX"

