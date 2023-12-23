class Field(object):
    """
    Represents a field within a dataset, holding information about transformations and data types.

    **Attributes:**
    - transform (callable, optional): A function to apply to values before processing.
    - dtype (str or numpy.dtype, optional): The desired data type for the field.
    """

    def __init__(self, transform=None, dtype=None):
        """
        Initializes a Field object.

        Args:
            transform (callable, optional): A function to apply to values.
            dtype (str or numpy.dtype, optional): The desired data type.
        """

        self.transform = transform
        self.dtype = dtype

    def process(self, value):
        """
        Processes a value according to the field's transformations and data type.

        Args:
            value: The value to process.

        Returns:
            The processed value.
        """

        # Apply transformation if specified
        if self.transform is not None:
            value = self.transform(value)

        # Convert to specified data type if provided
        if self.dtype is not None:
            # Note: astype() is a NumPy method, implying value is NumPy-compatible
            value = value.astype(self.dtype)

        return value


class Example(object):
    """
    Represents an example within a dataset, holding values for its fields.
    """

    @classmethod
    def fromlist(cls, values, fields):
        """
        Creates an Example object from a list of values and corresponding fields.

        Args:
            values: A list of values for the fields.
            fields: A list of field templates, each in the format (name, Field()).

        Returns:
            An Example object with processed values according to the fields.
        """

        example = cls()  # Create an instance of the Example class

        # Iterate through values and fields in parallel
        for (value, field) in zip(values, fields):
            # Assertions for field validity
            assert len(
                field) == 2, f"Expected a field template similar to ('name_field', Field()) but got {format(field)}"
            assert isinstance(
                field[1], Field), f"Expected a field template similar to ('name_field', Field()) but got {format(field)}"

            # Extract field name and process the value using the associated Field
            name = field[0]
            value = field[1].process(value)  # Apply processing from the Field object

            # Set the processed value as an attribute of the Example object
            setattr(example, name, value)

        return example
