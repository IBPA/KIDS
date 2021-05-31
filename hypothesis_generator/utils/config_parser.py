"""
Filename: config_parser.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Process .ini configuration files.

To-do:
"""
import configparser


class ConfigParser:
    """
    Config parser.
    """
    def __init__(self, filepath):
        """
        Class initializer for ConfigParser.

        Inputs:
            filepath: file path of .ini config file
        """
        self.config = configparser.ConfigParser()
        self.config.read(filepath)

    def append(self, section, entry):
        """
        Append to the configuration.

        Inputs:
            section: section name to append
            entry: items to append to specified section
                in dictionary format
        """
        self.config[section] = entry

    def write(self, filepath):
        """
        Write configuration.

        Inputs:
            filepath: file path to save the config file
        """
        self.config.write(filepath)

    def getstr(self, key, section='DEFAULT'):
        """
        Get key from configuration in string.

        Inputs:
            key: string for key to fetch
            section: (optional) section to fetch from

        Returns:
            configuration in string format
        """
        return self.config[section][key]

    def getint(self, key, section='DEFAULT'):
        """
        Get key from configuration in integer.

        Inputs:
            key: string for key to fetch
            section: (optional) section to fetch from

        Returns:
            configuration in integer format
        """
        return self.config.getint(section, key)

    def getbool(self, key, section='DEFAULT'):
        """
        Get key from configuration in boolean.

        Inputs:
            key: string for key to fetch
            section: (optional) section to fetch from

        Returns:
            configuration in boolean format
        """
        return self.config.getboolean(section, key)

    def getfloat(self, key, section='DEFAULT'):
        """
        Get key from configuration in float.

        Inputs:
            key: string for key to fetch
            section: (optional) section to fetch from

        Returns:
            configuration in float format
        """
        return self.config.getfloat(section, key)
