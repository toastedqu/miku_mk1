@ECHO OFF
call conda activate miku_vc
cd so-vits-svc-4.0
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -s "miku" -n "temp.wav" -t 0
call conda deactivate
exit