import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


class ParserError(Exception):
    """Base exception for parser errors."""
    pass


class SectionFormatError(ParserError):
    """Exception raised when section format is invalid."""
    pass


class MPIAllPrintFormatError(ParserError):
    """Exception raised when MPI ALL PRINT format is invalid."""
    pass


@dataclass
class ParserPatterns:
    """
    Configuration class for parser patterns.
    Modify these patterns to adapt the parser to different formats.
    """
    # Section patterns
    section_start: str = r'=\+=\+=\+= (?P<name>\S+) :: (?P<title>.+?) =\+=\+=\+='
    section_end: str = r'=\+=\+=\+= (?P<name>\S+) END =\+=\+=\+='
    
    # MPI ALL PRINT patterns
    mpi_start: str = r'-\+-\+-\+- (?P<name>.+?) -\+-\+-\+-'
    mpi_end: str = r'-\+-\+-\+- (?P<name>.+?) END -\+-\+-\+-'
    rank_block: str = r'\[\[Rank (?P<rank>\d+)\]\]\n(?P<content>.*?)\n\[\[END Rank (?P=rank)\]\]'
    
    def __post_init__(self):
        """Compile all patterns for efficiency."""
        self.section_start_re = re.compile(self.section_start)
        self.section_end_re = re.compile(self.section_end)
        self.mpi_start_re = re.compile(self.mpi_start)
        self.mpi_end_re = re.compile(self.mpi_end)
        self.rank_block_re = re.compile(self.rank_block, re.DOTALL)


@dataclass
class MPIAllPrint:
    """Represents a parsed MPI ALL PRINT block."""
    name: str
    rank_outputs: Dict[int, str] = field(default_factory=dict)
    raw_text: str = ""
    
    def get_rank_output(self, rank: int) -> Optional[str]:
        """Get output for a specific rank."""
        return self.rank_outputs.get(rank)
    
    def get_all_ranks(self) -> List[int]:
        """Get list of all available ranks."""
        return sorted(self.rank_outputs.keys())


@dataclass
class Section:
    """Represents a parsed section."""
    name: str
    title: str
    raw_text: str
    mpi_all_prints: Dict[str, MPIAllPrint] = field(default_factory=dict)
    
    def get_mpi_print(self, name: str) -> Optional[MPIAllPrint]:
        """Get an MPI ALL PRINT by name."""
        return self.mpi_all_prints.get(name)
    
    def list_mpi_prints(self) -> List[str]:
        """Get list of all MPI ALL PRINT names."""
        return list(self.mpi_all_prints.keys())


class MPIOutputParser:
    """
    Parser for MPI output files with sections and all-print blocks.
    
    Example usage:
        parser = MPIOutputParser()
        result = parser.parse_file('output.txt')
        
        # Access sections
        section = result['initialization']
        print(section.title)
        
        # Access MPI prints
        mpi_print = section.get_mpi_print('setup_info')
        print(mpi_print.get_rank_output(0))
    """
    
    def __init__(self, patterns: Optional[ParserPatterns] = None):
        """
        Initialize parser with optional custom patterns.
        
        Args:
            patterns: Custom ParserPatterns instance. If None, uses defaults.
        """
        self.patterns = patterns or ParserPatterns()
    
    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, Section]:
        """
        Parse an MPI output file.
        
        Args:
            filepath: Path to the file to parse
            
        Returns:
            Dictionary mapping section names to Section objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ParserError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_string(content)
    
    def parse_string(self, content: str) -> Dict[str, Section]:
        """
        Parse MPI output from a string.
        
        Args:
            content: String content to parse
            
        Returns:
            Dictionary mapping section names to Section objects
            
        Raises:
            ParserError: If content format is invalid
        """
        sections = {}
        pos = 0
        
        while pos < len(content):
            # Find next section start
            match = self.patterns.section_start_re.search(content, pos)
            if not match:
                # No more sections
                break
            
            section_name = match.group('name')
            section_title = match.group('title').strip()
            section_start_pos = match.end()
            
            # Find corresponding section end
            end_match = self.patterns.section_end_re.search(content, section_start_pos)
            if not end_match:
                raise SectionFormatError(
                    f"Section '{section_name}' starting at position {match.start()} "
                    "has no matching END marker"
                )
            
            end_name = end_match.group('name')
            if end_name != section_name:
                raise SectionFormatError(
                    f"Section end name mismatch: expected '{section_name}', "
                    f"got '{end_name}' at position {end_match.start()}"
                )
            
            # Extract section content
            section_content = content[section_start_pos:end_match.start()]
            
            # Parse MPI ALL PRINTs within this section and get unparsed content
            mpi_prints, raw_text = self._parse_mpi_all_prints(section_content, section_name)
            
            # Create section object
            section = Section(
                name=section_name,
                title=section_title,
                raw_text=raw_text,
                mpi_all_prints=mpi_prints
            )
            
            if section_name in sections:
                raise SectionFormatError(
                    f"Duplicate section name '{section_name}' found"
                )
            
            sections[section_name] = section
            pos = end_match.end()
        
        return sections
    
    def _parse_mpi_all_prints(self, content: str, section_name: str) -> tuple[Dict[str, MPIAllPrint], str]:
        """
        Parse all MPI ALL PRINT blocks within a section.
        
        Args:
            content: Section content to parse
            section_name: Name of the parent section (for error messages)
            
        Returns:
            Tuple of (mpi_prints dict, raw_text with MPI blocks removed)
        """
        mpi_prints = {}
        
        # Track all MPI ALL PRINT blocks to remove them from raw text
        mpi_blocks_to_remove = []
        
        pos = 0
        while pos < len(content):
            # Find next MPI ALL PRINT start
            match = self.patterns.mpi_start_re.search(content, pos)
            if not match:
                break
            
            mpi_name = match.group('name').strip()
            mpi_start_pos = match.start()  # Include the start marker
            
            # Find corresponding MPI ALL PRINT end
            end_match = self.patterns.mpi_end_re.search(content, match.end())
            if not end_match:
                raise MPIAllPrintFormatError(
                    f"MPI ALL PRINT '{mpi_name}' in section '{section_name}' "
                    f"starting at position {match.start()} has no matching END marker"
                )
            
            end_name = end_match.group('name').strip()
            if end_name != mpi_name:
                raise MPIAllPrintFormatError(
                    f"MPI ALL PRINT end name mismatch in section '{section_name}': "
                    f"expected '{mpi_name}', got '{end_name}' at position {end_match.start()}"
                )
            
            # Extract MPI print content (between start and end markers)
            mpi_content = content[match.end():end_match.start()]
            
            # Track the entire block including markers for removal
            mpi_end_pos = end_match.end()
            mpi_blocks_to_remove.append((mpi_start_pos, mpi_end_pos))
            
            # Parse rank blocks
            rank_outputs = self._parse_rank_blocks(mpi_content, mpi_name, section_name)
            
            # Create MPI ALL PRINT object
            mpi_print = MPIAllPrint(
                name=mpi_name,
                rank_outputs=rank_outputs,
                raw_text=mpi_content
            )
            
            if mpi_name in mpi_prints:
                raise MPIAllPrintFormatError(
                    f"Duplicate MPI ALL PRINT name '{mpi_name}' in section '{section_name}'"
                )
            
            mpi_prints[mpi_name] = mpi_print
            pos = end_match.end()
        
        # Remove MPI blocks from content to get raw text
        raw_text = content
        # Remove blocks in reverse order to maintain correct positions
        for start, end in reversed(mpi_blocks_to_remove):
            raw_text = raw_text[:start] + raw_text[end:]
        
        # Clean up extra whitespace
        raw_text = raw_text.strip()
        
        return mpi_prints, raw_text
    
    def _parse_rank_blocks(self, content: str, mpi_name: str, section_name: str) -> Dict[int, str]:
        """
        Parse rank blocks within an MPI ALL PRINT.
        
        Args:
            content: MPI ALL PRINT content
            mpi_name: Name of the MPI print (for error messages)
            section_name: Name of the parent section (for error messages)
            
        Returns:
            Dictionary mapping rank numbers to their output strings
        """
        rank_outputs = {}
        
        for match in self.patterns.rank_block_re.finditer(content):
            rank = int(match.group('rank'))
            output = match.group('content')
            
            if rank in rank_outputs:
                raise MPIAllPrintFormatError(
                    f"Duplicate rank {rank} in MPI ALL PRINT '{mpi_name}' "
                    f"of section '{section_name}'"
                )
            
            rank_outputs[rank] = output
        
        return rank_outputs
    
    

def parse_ccutils_output_file(path: Path) -> Dict[str, Section]:
    """
    Convenience function to parse CCUTILS output.
    
    Args:
        source: File path to parse
        
    Returns:
        Dictionary mapping section names to Section objects
    """
    parser = MPIOutputParser()
    return parser.parse_file(path)
     

def parse_ccutils_output(text: str) -> Dict[str, Section]:
    """
    Convenience function to parse CCUTILS output.
    
    Args:
        text: String content to parse
        
    Returns:
        Dictionary mapping section names to Section objects
    """
    parser = MPIOutputParser()
    return parser.parse_string(str(text))