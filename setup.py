from setuptools import setup, find_packages

setup(
    name='testing-harness',
    version='0.1',
    description='ESnet Testing Harness',
    url='https://github.com/esnet/testing-harness',
    author='Ezra Kissel',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='testing harness protocol iperf3 perfsonar',

    packages=find_packages(),

    install_requires=['requests', 'pyyaml', 'pika', 'ping3', 'tabulate', 'pptx']
)
