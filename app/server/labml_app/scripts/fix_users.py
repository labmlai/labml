import time

from labml import monit
from labml_app.db import init_db, user, app_token, password_reset
from labml_app.utils import emails, gravatar_utils


def _gen_password_reset_email(us: user.User, reset_url: str):
    return f"""
<html style="scroll-behavior: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; touch-action: manipulation;">
<body class="light"
      style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; touch-action: manipulation; font-size: 1.1rem;">
<div style="margin-top: 50px;">
    <div style="margin: 0 auto; width:500px; background-color: #ecf0f3; padding: 25px; border-radius: 25px;">
        <h4>Hi {us.name},</h4> <p> Due to a migration in our hosted <a href="https://app.labml.ai">app</a> you will 
        have to reset your password. Please click or paste the following URL into your web browser to reset your 
        password. </p> <p>{reset_url} </p> 
        <div style="margin-top: 50px; text-align:center;">
            <span><a href="contact@labml.ais">Contact Us</a> | <a href="contact@labml.ai">Privacy Policy</a> | <a
                    href="https://app.labml.ai">Papers</a></span>
        </div>
    </div>
</div>
</body>
</html>
    """


init_db()

app_tokens = app_token.AppToken.get_all()
for app_token_key in monit.iterate('App Tokens', app_tokens):
    at: app_token.AppToken = app_token_key.load()
    if at is not None and at.user is not None:
        u = at.user.load()

        u.tokens[at.token_id] = at.expiration
        u.session_token_owners[at.token_id] = set()
        user.UserTokenIndex.set(at.token_id, u.key)
        u.save()

users = user.User.get_all()
for user_key in monit.iterate('Users', users):
    u: user.User = user_key.load()
    u.picture = gravatar_utils.get_image_url(u.email)
    u.save()

    user.UserEmailIndex.set(u.email, u.key)

    reset_token = password_reset.gen_reset_token(u.email)
    html = _gen_password_reset_email(u, f"https://app.labml.ai/auth/reset?token={reset_token}")
    email_client = emails.Email([u.email])
    email_client.send('Please Reset Your Password', html)
