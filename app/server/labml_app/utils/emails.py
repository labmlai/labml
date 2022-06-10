import time
from typing import List

import boto3
from botocore.exceptions import ClientError
from labml import logger

from labml_app.settings import EMAIL_SENDER, EMAIL_AWS_REGION, EMAIL_CHARSET


class Email:
    sender_email: str
    recipient_emails: List[str]
    client: boto3.client

    def __init__(self, recipient_emails: List[str], sender_email: str = EMAIL_SENDER):
        self.recipient_emails = recipient_emails
        self.sender_email = sender_email
        if EMAIL_AWS_REGION:
            self.client = boto3.client('ses', region_name=EMAIL_AWS_REGION)

    def _send(self, subject: str, body_html: str, body_text: str = '') -> List[str]:
        res = []
        if not self.client:
            return res

        for recipient_email in self.recipient_emails:
            try:
                response = self.client.send_email(
                    Destination={
                        'ToAddresses': [recipient_email]
                    },
                    Message={
                        'Body': {
                            'Html': {
                                'Charset': EMAIL_CHARSET,
                                'Data': body_html,
                            },
                            'Text': {
                                'Charset': EMAIL_CHARSET,
                                'Data': body_text,
                            },
                        },
                        'Subject': {
                            'Charset': EMAIL_CHARSET,
                            'Data': subject,
                        },
                    },
                    ReplyToAddresses=[EMAIL_SENDER],
                    ReturnPath=self.sender_email,
                    Source=self.sender_email,
                )
                time.sleep(1)
                logger.log(f'sent to {recipient_email}')
            except ClientError as e:
                # raise EmailFailed(f"{e.response['Error']['Message']}, email:{recipient_email}")
                logger.log(f"{e.response['Error']['Message']}, email:{recipient_email}")
            else:
                res.append(response['MessageId'])

        return res

    def send(self, subject: str, body_html: str, body_text: str = '') -> List[str]:
        return self._send(subject, body_html, body_text)
