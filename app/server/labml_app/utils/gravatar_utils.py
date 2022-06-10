try:
    from libgravatar import Gravatar
except ImportError:
    pass


def get_image_url(email: str) -> str:
    g = Gravatar(email)
    return g.get_image(
        default='https://papers.labml.ai/images/user.jpg')
