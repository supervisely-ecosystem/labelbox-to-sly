from src.ui.info import info_text

try:
    from labelbox.exceptions import ApiLimitError, AuthenticationError
except ImportError:
    from lbox.exceptions import AuthenticationError, ApiLimitError


def handle_lb_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            info_text.text = "Invalid API KEY. Please check it in the settings of the app."
            info_text.status = "error"
            info_text.show()
            raise Exception(f"Invalid API KEY. Please check it in the settings of the app.")
        except ApiLimitError as e:
            info_text.text = (
                "Your API KEY for Labelbox has limitations. Please contact our support team."
            )
            info_text.status = "error"
            info_text.show()
            raise Exception(
                "Your API KEY for Labelbox has limitations. Please contact our support team."
            )
        except Exception as e:
            info_text.text = (
                f"Someting went wrong. Check logs for more info and contact our support team."
            )
            info_text.status = "error"
            info_text.show()
            raise Exception(f"Someting went wrong. Please contact our support team. {str(e)}")

    return wrapper
