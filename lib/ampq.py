import logging
import pika


log = logging.getLogger("harness")


class AMPQSender(object):

    def __init__(self, url, queue, persistent=False):
        self._url = url
        try:
            self._conn = pika.BlockingConnection(
                pika.ConnectionParameters(host=url))
        except Exception as e:
            log.error(f"Could not connect to {self._url}: {e}")
            return

        self._channel = self._conn.channel()

        self._channel.queue_declare(queue=queue, durable=True)
        self._channel.confirm_delivery()  # do we want this?
        self._persist = persistent

    def send(self, key, jstr):
        try:
            log.debug(f"Sending data ({len(jstr)} items) to archive {self._url}\n")
            try:
                self._channel.basic_publish(exchange='',
                                            routing_key=key,
                                            body=jstr)
            except pika.exceptions.UnroutableError:
                print('Message was returned')

            if not self._persist:
                self._conn.close()
        except Exception as e:
            log.error(f"Could not send ampq to {self._url}: {e}")
