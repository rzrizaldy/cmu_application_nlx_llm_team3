from pydantic import BaseModel


class Pittsburgh311Response(BaseModel):
    request_type_id: str
    issue: str
    category: str
    department: str