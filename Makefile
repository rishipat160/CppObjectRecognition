CC = cl
CFLAGS = /MD /EHsc
INCLUDES = /I "include" /I "..\opencv\build\include"
LIBPATH = /link /LIBPATH:"..\opencv\build\x64\vc16\lib" 
LIBS = opencv_world4110.lib 
SRCDIR = src

vidDisplay:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/vidDisplay.cpp $(SRCDIR)/threshold.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

runVid: vidDisplay
	.\bin\vidDisplay.exe

clean:
	del bin\*.obj bin\*.exe *.jpg 