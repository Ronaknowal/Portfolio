const imports = {
  "typed-decision-model": () => import("./typed-decision-model/content.jsx"),
};
const requests = new Map();

export function loadProject(id) {
  if (!Object.hasOwn(imports, id)) return Promise.reject(new Error("Unknown project"));
  if (!requests.has(id)) {
    const request = imports[id]()
      .then(module => {
        if (!module.default || typeof module.default !== "object") {
          throw new Error("Project module has no stage content");
        }
        return module.default;
      })
      .catch(error => {
        requests.delete(id);
        throw error;
      });
    requests.set(id, request);
  }
  return requests.get(id);
}
