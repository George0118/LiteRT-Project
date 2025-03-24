@echo off

cd ./testing

for /L %%i in (0, 1, 9) do (
    python main.py model_%%i > ./results/results_%%i.txt
)

echo Script execution complete.
