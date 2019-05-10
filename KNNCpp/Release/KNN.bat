@echo off

for /l %%k in (1,1,30) do (
	for /l %%i in (1,1,100) do (
		echo k = %%k i = %%i
		call KNNCpp.exe titanic.dat %%k >> result.txt
	)
)

pause
