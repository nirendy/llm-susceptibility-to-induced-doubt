import sys
from contextlib import contextmanager
from io import StringIO
from threading import current_thread
from typing import Callable
from typing import Generic
from typing import TypeVar

import streamlit as st

TSessionKey = TypeVar("TSessionKey")


class SessionKey(Generic[TSessionKey]):
    def __init__(self, key):
        self.key = key

    @staticmethod
    def with_default(key, default_value: TSessionKey) -> "SessionKey[TSessionKey]":
        session_key = SessionKey(key)
        session_key.init(default_value)
        return session_key

    def exists(self) -> bool:
        return self.key in st.session_state

    def delete(self):
        if self.exists():
            del st.session_state[self.key]

    def update(self, value: TSessionKey):
        st.session_state[self.key] = value

    def init(self, value: TSessionKey):
        if not self.exists():
            st.session_state[self.key] = value

    def get(self) -> TSessionKey:
        return st.session_state[self.key]

    def equal_if_exists(self, func: Callable[[TSessionKey], bool]) -> bool:
        if self.exists():
            return func(self.get())
        return False

    def exists_and_not_none(self) -> bool:
        return self.equal_if_exists(lambda val: val is not None)

    def update_button(self, value: TSessionKey, label):
        st.button(label=label, key=label, on_click=lambda: self.update(value))


@contextmanager
def st_redirect(src, dst, placeholder, overwrite):
    output_func = getattr(placeholder.empty(), dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):

            is_newline = b == "\n"
            if is_newline:
                return

            old_write(b)
            buffer.write(b + "\r\n")

            # Without this condition, will cause infinite loop because we can't write to the streamlit from thread
            if getattr(current_thread(), st.script_run_context.SCRIPT_RUN_CONTEXT_ATTR_NAME, None) is None:
                if overwrite:
                    buffer.truncate(0)
                    buffer.seek(0)
                return

            output_func(buffer.getvalue())

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst, placeholder, overwrite):
    "this will show the prints"
    with st_redirect(sys.stdout, dst, placeholder, overwrite):
        yield


@contextmanager
def st_stderr(dst, placeholder, overwrite):
    "This will show the logging"
    with st_redirect(sys.stderr, dst, placeholder, overwrite):
        yield
