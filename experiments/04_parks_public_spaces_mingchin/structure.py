'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from pathlib import Path
import json

from src.pydantic_models.pittsburgh311 import Pittsburgh311Response


output_dir = Path("data/pydantic_models")
output_dir.mkdir(parents=True, exist_ok=True)

schema = Pittsburgh311Response.model_json_schema()

output_file = output_dir / "pittsburgh311_schema.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

print("Saved schema to:", output_file)
