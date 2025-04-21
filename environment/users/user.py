class User:
    """
    Object to represent users that will interacts with the s
    """

    id_counter = 0

    def __init__(self, name, description):
        """
        id (integer): unique identifier for each user
        name (string): name + surname of the user
        description (string): small description of the user, including its cinematic interests
        """
        self.id = User.id_counter
        User.id_counter += 1
        self.name = name
        self.description = description

    def __str__(self) -> str:
        return (
            f"User(id = {self.id}, name = {self.name}\n{self.description}))"
        )

    def __repr__(self) -> str:
        return f"User(id = {self.id}, name = {self.name})"

    @staticmethod
    def get_num_users():
        return User.id_counter

