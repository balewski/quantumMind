FROM 292282985366.dkr.ecr.us-east-1.amazonaws.com/amazon-braket-base-jobs:latest

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz \
        && mkdir /usr/local/julia \
        && tar xf julia-1.8.3-linux-x86_64.tar.gz --directory /usr/local/julia --strip-components=1 \
        && ln -s /usr/local/julia/bin/julia /usr/local/bin/julia \
        && rm -f julia-1.8.3-linux-x86_64.tar.gz

RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.add(url = "https://github.com/amazon-braket/BraketAHS.jl.git", rev = "1372d6cd1fec7d63f8849077446ceff447a06394"); Pkg.add("JSON3"); Pkg.precompile()'

RUN pip install --upgrade amazon-braket-sdk
RUN pip install --upgrade git+https://github.com/amazon-braket/amazon-braket-default-simulator-python.git@99120621089772fb467cc8168e2d40538acfe98e