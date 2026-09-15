import os
import re
import subprocess
import tempfile
from pathlib import Path

from conceptflow.visualization.dimflux.src.utils.variables import Variables


def _match_known_tokens(text: str, vocabulary: set) -> set:
    '''
    Segment a whitespace-joined string back into a set of known names.

    The JAR prints Clojure set elements separated by plain spaces, with no
    delimiter to tell apart a multi-word name (e.g. "lives in water") from
    several single-word names in a row. Since the full set of valid names is
    already known ahead of time, greedy longest-match word segmentation
    against that vocabulary recovers the original names unambiguously for
    any non-adversarial (non self-overlapping) label set.
    '''
    words = text.split()
    matched: set = set()
    i = 0
    n = len(words)

    while i < n:
        for length in range(n - i, 0, -1):
            candidate = " ".join(words[i:i + length])
            if candidate in vocabulary:
                matched.add(candidate)
                i += length
                break
        else:
            raise ValueError(
                f"Could not segment {' '.join(words[i:])!r} into known names "
                f"from {sorted(vocabulary)}."
            )

    return matched


class DimDraw():
    '''
    Disclaimer
    ----------
    The original version is integrated into the tool conexp-clj
    [see https://github.com/tomhanika/conexp-clj].

    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''
    def __init__(self,
            variables: Variables
        ):
        '''
        Initialize DimDraw with a given Variables object.

        Parameters
        ----------
        variables: Variables
            The storage of variables.
        '''
        self.vars = variables
        self.concepts = {}

        for concept in self.vars.concepts:
            extent = self.vars.extents[concept]
            intent = self.vars.intents[concept]

            extent_names = {
                self.vars.objects[index] if isinstance(index, int) else index
                for index in extent
            }

            intent_names = {
                self.vars.attributes[index] if isinstance(index, int) else index
                for index in intent
            }

            self.concepts[concept] = extent_names | intent_names

    def two_dimensional_extension(self):
        '''
        Compute the two-dimensional extension of the lattice.

        Raises
        ------
        ValueError
            If the script cannot find or run the provided JAR-file of brunt.

        Returns
        -------
        realizer : list
            Two-dimensional extension of the concept lattice.
        '''
        le_x, le_y = [None] * self.vars.N_c, [None] * self.vars.N_c

        java_bin = os.environ.get("CONCEPTFLOW_JAVA_BIN", "java")
        try:
            base_dir = Path(__file__).resolve().parents[2]
            jar_path = base_dir / "libs" / "brunt-fork.jar"

            # Always work from the ConceptFlow FormalContext object.
            # If the original input was already a .cxt file, use it directly.
            # Otherwise, create a temporary .cxt export for the jar.
            if str(self.vars.cxt).endswith('.cxt'):
                cxt_path = Path(self.vars.cxt).resolve()
                res = subprocess.check_output(
                    [java_bin, "-jar", str(jar_path), "-f", "dim-draw-coordinates", str(cxt_path)],
                    text=True,
                    stderr=subprocess.STDOUT
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=".cxt", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    self.vars.context.to_cxt(tmp_path)
                    res = subprocess.check_output(
                        [java_bin, "-jar", str(jar_path), "-f", "dim-draw-coordinates",
                         str(cxt_path)],
                        text=True,
                        stderr=subprocess.STDOUT
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

            for line in res.splitlines():
                concept, coords = line.split(' -> ')
                x, y = coords.strip("()").split(", ")
                # JAR outputs Clojure set literals: [#{obj1 obj2 ...} #{attr1 ...}]
                # Parse both sets and union them to match the self.concepts dict.
                # Elements are whitespace-joined with no other delimiter, so a
                # blind str.split() breaks the moment an object/attribute name
                # itself contains a space (e.g. "lives in water"). Since the
                # full name vocabulary is already known (self.vars.G/M), greedy
                # longest-match segmentation against it recovers the original
                # multi-word names correctly.
                groups = re.findall(r'#\{([^}]*)\}', concept)
                extent_text = groups[0] if groups else ""
                intent_text = groups[1] if len(groups) > 1 else ""
                extent_tokens = _match_known_tokens(extent_text, self.vars.G)
                intent_tokens = _match_known_tokens(intent_text, self.vars.M)
                concept_elements = extent_tokens | intent_tokens

                node = next(
                    (k for k, v in self.concepts.items() if v == concept_elements),
                    None
                )

                if node is None:
                    raise ValueError(
                        f"Could not match DimDraw concept output to a "
                        f"ConceptFlow concept: {concept}"
                    )

                le_x[int(x)] = node
                le_y[int(y)] = node

            self.realizer = [le_x, le_y]
            return self.realizer

        except subprocess.CalledProcessError as e:
            raise ValueError(f'Error running JAR: {e}')
