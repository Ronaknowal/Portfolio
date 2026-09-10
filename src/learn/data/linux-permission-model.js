// An existing regular file, with ordinary mode bits and no symlinks. The caller
// is a non-owner member of the owning group on both objects. The fixed / and
// /project ancestors can be searched. ACLs and privilege overrides are absent.
export const initialPathPermissions = Object.freeze({
  directoryRead: true,
  directorySearch: false,
  fileRead: true,
});

export function pathPermissionModel({ directoryRead, directorySearch, fileRead }) {
  const blockedAt = !directorySearch ? "directory-search" : !fileRead ? "file-read" : null;
  return {
    canListNames: directoryRead,
    canReadFile: directorySearch && fileRead,
    fileReadChecked: directorySearch,
    blockedAt,
    directoryGroupDigit: (directoryRead ? 4 : 0) + (directorySearch ? 1 : 0),
    fileGroupDigit: fileRead ? 4 : 0,
    gates: {
      ancestors: "passed",
      directory: directorySearch ? "passed" : "blocked",
      file: !directorySearch ? "not-reached" : fileRead ? "passed" : "blocked",
    },
  };
}
