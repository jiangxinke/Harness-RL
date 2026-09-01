export function isSPActionToolName(toolName: string): boolean {
  return toolName.startsWith("sp_") && toolName.length > 3;
}

export function getSPActionType(toolName: string): string {
  return isSPActionToolName(toolName)
    ? toolName.slice(3).toUpperCase()
    : toolName.toUpperCase();
}
