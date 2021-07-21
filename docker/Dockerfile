# perfSONAR Testpoint with tool updates and ESnet Testing Harness included

# Build in temporary image
FROM perfsonar/testpoint:latest AS build
WORKDIR /src

ARG IP_RT2_VER=5.8.0

RUN yum -y install git wget
RUN yum -y groupinstall "Development tools"

# Get updated version of ss tool
RUN wget https://git.kernel.org/pub/scm/network/iproute2/iproute2.git/snapshot/iproute2-${IP_RT2_VER}.tar.gz 
RUN tar xvzf iproute2-${IP_RT2_VER}.tar.gz; cd iproute2-${IP_RT2_VER}; ./configure && make DESTDIR=/usr/local -j4 && make DESTDIR=/usr/local install

# Get updated verion of mpstat that support JSON output
RUN git clone https://github.com/sysstat/sysstat.git
RUN cd sysstat && ./configure --prefix=/usr/local && make -j4 && make install

# Get iperf3 version that does different congestion control on even/odd numbered streams
RUN git clone -b bbr2-testing https://github.com/esnet/iperf.git
RUN cd iperf && ./configure --enable-static --disable-shared && make -j4

# Now build final perfSONAR extended image
FROM perfsonar/testpoint:latest
WORKDIR /src

RUN yum -y install git wget tcsh

# Move over binaries and configuratioon from build image
COPY --from=build /usr/local/ /usr/local/
COPY --from=build /src/iperf/src/iperf3 /usr/bin/iperf3-cc-hack

# Create a symlink iperf3-dist for clarity
RUN ln -s /usr/bin/iperf3 /usr/bin/iperf3-dist

# Fix problem running tcpdump in priviledged container
RUN mv /usr/sbin/tcpdump /usr/bin/tcpdump; ln -s /usr/bin/tcpdump /usr/sbin/tcpdump

# Get and configure testing harness
RUN git clone https://github.com/esnet/testing-harness.git
RUN cd testing-harness && pip3 install .
RUN ln -s /src/testing-harness/ /harness
