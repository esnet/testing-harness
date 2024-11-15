# perfSONAR Testpoint with tool updates and ESnet Testing Harness included

# Build in temporary image
FROM perfsonar/testpoint:latest AS build
# systemd version not working??
#FROM perfsonar/testpoint:v4.4.4-systemd AS build
WORKDIR /src

ARG IP_RT2_VER=6.4.0
#ARG IP_RT2_VER=5.8.0

RUN yum -y install git wget tcsh ethtool openssl-devel
RUN yum -y groupinstall "Development tools" 

# Get updated version of ss tool
RUN wget --no-check-certificate https://git.kernel.org/pub/scm/network/iproute2/iproute2.git/snapshot/iproute2-${IP_RT2_VER}.tar.gz 
RUN tar xvzf iproute2-${IP_RT2_VER}.tar.gz; cd iproute2-${IP_RT2_VER}; ./configure && make DESTDIR=/usr/local -j4 && make DESTDIR=/usr/local install

# Get updated verion of mpstat that support JSON output
RUN git clone https://github.com/sysstat/sysstat.git
RUN cd sysstat && ./configure --prefix=/usr/local && make -j4 && make install

# install threaded version of iperf3 in /usr
RUN wget https://github.com/esnet/iperf/archive/refs/tags/iperf-3.15-mt-beta1.tar.gz
RUN tar xvzf iperf-3.15-mt-beta1.tar.gz
RUN cd iperf-* && ./configure --prefix=/usr && make && make install

# install modified version of iperf3 that does different congestion control on even/odd numbered streams
# install in /usr/local
#RUN git clone -b bbr2-testing https://github.com/esnet/iperf.git
RUN git clone -b mt https://github.com/bltierney/iperf3-bbr-testing.git
RUN cd iperf3-bbr-testing && ./configure --prefix=/usr/local && make -j4 && make install

# Now build final perfSONAR extended image
FROM perfsonar/testpoint:latest
#FROM perfsonar/testpoint:v4.4.4-systemd
WORKDIR /src

RUN yum -y install git wget tcsh ethtool openssl-devel
RUN yum -y groupinstall "Development tools" 

# Move over binaries and configuratioon from build image
COPY --from=build /usr/local/ /usr/local/
COPY --from=build /src/iperf3-bbr-testing/src/iperf3 /usr/bin/iperf3-cc-hack

# Create a symlink iperf3-dist for clarity
RUN ln -s /usr/bin/iperf3 /usr/bin/iperf3-dist

# Fix problem running tcpdump in priviledged container
RUN mv /usr/sbin/tcpdump /usr/bin/tcpdump; ln -s /usr/bin/tcpdump /usr/sbin/tcpdump

# Get and configure testing harness
RUN git clone -b bbr3-testing https://github.com/esnet/testing-harness.git
# this runs setup.py
RUN cd testing-harness && pip3 install .
RUN ln -s /src/testing-harness/ /harness

# Needed to run iperf3-mt for now 
ENV LD_LIBRARY_PATH=/usr/bin:$LD_LIBRARY_PATH

# share with host OS
VOLUME /data
