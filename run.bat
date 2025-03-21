@echo off
SET EXE_PATH=C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project\testing\build\Release\TFLiteCheck.exe
SET MODEL_PATH=C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project\models\model_result\model_
SET RESULTS_PATH=C:\Users\gmyst\Desktop\Work\Delft\LiteRT-Project\testing\results\results_

REM Enable delayed variable expansion
SETLOCAL ENABLEDELAYEDEXPANSION

REM Loop through model_0 to model_9 and run the executable with the corresponding argument
FOR /L %%i IN (0, 1, 9) DO (
    SET MODEL_FILE=!MODEL_PATH!%%i\converted_model.tflite
    SET RESULT_FILE=!RESULTS_PATH!%%i.txt

    REM Run the executable and redirect output to the result file
    echo Running TFLiteCheck.exe with argument: !MODEL_FILE!
    "%EXE_PATH%" "!MODEL_FILE!" > "!RESULT_FILE!"

    REM Optional: Print message when done with each model
    echo Output saved to !RESULT_FILE!
)

REM End delayed expansion
ENDLOCAL

pause
