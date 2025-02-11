import ctypes
from ctypes import wintypes


def open_save_dialog_native():
    """
    Opens a native Windows save file dialog and returns the selected file path.

    Returns:
        str: The file path chosen by the user, or an empty string if canceled.
    """

    # Define the OPENFILENAME structure
    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    # Prepare the buffer for the file path
    file_buffer = ctypes.create_unicode_buffer(260)  # MAX_PATH = 260

    # Set up the dialog parameters
    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.lpstrFilter = "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0"
    # ofn.lpstrFile = file_buffer
    ofn.lpstrFile = ctypes.cast(file_buffer, wintypes.LPWSTR)
    ofn.nMaxFile = 260
    ofn.lpstrDefExt = "txt"
    ofn.lpstrTitle = "Save Your File"
    ofn.Flags = 0x00000002 | 0x00000004  # OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY

    # Call the native Save File Dialog
    if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
        return file_buffer.value  # Return the selected file path
    else:
        return ""  # If canceled, return an empty string


def open_file_dialog_native():
    """
    Opens a native Windows file open dialog and returns the selected file path.

    Returns:
        str: The file path chosen by the user, or an empty string if canceled.
    """

    # Define the OPENFILENAME structure
    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    # Prepare the buffer for the file path
    file_buffer = ctypes.create_unicode_buffer(260)  # MAX_PATH = 260

    # Set up the dialog parameters
    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    # ofn.lpstrFilter = "Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0"
    ofn.lpstrFile = ctypes.cast(file_buffer, wintypes.LPWSTR)  # Cast buffer to LPWSTR
    ofn.nMaxFile = 260
    ofn.lpstrTitle = "Select a File"
    ofn.Flags = 0x00000008 | 0x00001000  # OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST

    # Call the native Open File Dialog
    if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return file_buffer.value  # Return the selected file path
    else:
        return ""  # If canceled, return an empty string


if __name__ == "__main__":
    # Example usage
    file_path = open_save_dialog_native()
    # file_path = open_file_dialog_native()
    if file_path:
        print(f"File will be saved to: {file_path}")
    else:
        print("Save operation canceled.")
