NVCC = nvcc

SOURCEDIR = src
SOURCECOMMONSDIR = src/commons
BINDIR = bin
TESTDIR = test

XFLAGS = /openmp
NVCCFLAGS = -arch=sm_60 -rdc=true

TARGETS = cpu_sequential cpu_openmp gpu_percube gpu_batch

TESTSCRIPT = test.py
RESULTSCRIPT = results.py

TESTOUTDIRS = test\build test\old test\runs
TESTOUTFILES = test\results.csv test\results.log

all : clean build

build : makedir $(TARGETS)

test ::
  cd $(TESTDIR) && python $(TESTSCRIPT) && python $(RESULTSCRIPT)

makedir :
  if not exist $(BINDIR) mkdir $(BINDIR) 

clean :
  if exist $(BINDIR) rmdir /S /Q $(BINDIR)

cleantest :
  (for %%a in ($(TESTOUTDIRS)) do (if exist "%%~a" rmdir /S /Q "%%~a")) & (for %%a in ($(TESTOUTFILES)) do (if exist "%%~a" del "%%~a"))

$(TARGETS) : functions
  $(NVCC) -Xcompiler="$(XFLAGS)" $(NVCCFLAGS) $(BINDIR)/functions.obj $(SOURCEDIR)/$@.cu -o $(BINDIR)/$@.exe

functions :
  $(NVCC) -c $(NVCCFLAGS) $(SOURCECOMMONSDIR)/$@.cu -o $(BINDIR)/$@.obj
