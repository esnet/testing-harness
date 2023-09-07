#!/bin/bash
# Build a bbrv3 Linux kernel, install it on the local machine
# based on Google script gce_install.sh
#
# See: https://github.com/google/bbr/blob/v3/README.md
#

set -e
set -x

git clone -o google-bbr -b v3  https://github.com/google/bbr.git
cd bbr/

KERNEL='6.4.0'  # see Makefile for version number
BRANCH=`git rev-parse --abbrev-ref HEAD | sed s/-/+/g`
SHA1=`git rev-parse --short HEAD`
LOCALVERSION=+${BRANCH}+${SHA1}
INSTALL_DIR=${PWD}/${LOCALVERSION}/install
BUILD_DIR=${PWD}/${LOCALVERSION}/build
MAKE_OPTS="-j`nproc` \
           LOCALVERSION=${LOCALVERSION} \
           EXTRAVERSION="" \
           INSTALL_PATH=${INSTALL_DIR}/boot \
           INSTALL_MOD_PATH=${INSTALL_DIR}"

echo "cleaning..."
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}/boot


# will base config on current working kernel
echo "running make localmodconfig ..."
rm .config
make localmodconfig

make ${MAKE_OPTS} prepare         > /tmp/make.prepare
echo "making..."
make ${MAKE_OPTS}                 > /tmp/make.default
echo "making modules ..."
make ${MAKE_OPTS} modules         > /tmp/make.modules
echo "making install ..."
make ${MAKE_OPTS} install         > /tmp/make.install
echo "making modules_install ..."
make ${MAKE_OPTS} modules_install > /tmp/make.modules_install
set -e

rm -rf "/boot/6.4.0+*v3*" "/lib/modules/*6.4.0+v3*"
cd ${INSTALL_DIR}
echo "copying /boot and /lib files ..."
cp boot/* /boot
cp -r lib/modules/* /lib/modules
cd /boot

#mkinitramfs -k -o initrd.img-${KERNEL}${LOCALVERSION} ${KERNEL}${LOCALVERSION}
#or could also do:
update-initramfs -u -k ${KERNEL}${LOCALVERSION}

update-grub
