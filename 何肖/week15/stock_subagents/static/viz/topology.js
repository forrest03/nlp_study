/* =============================================================================
 *  topology.js  ——  拓扑可视化（辅助代码，非教学重点）· 深色科技感 · 统一状态配色
 * =============================================================================
 *  vanilla JS + SVG，零依赖。学生不需要读懂这里。
 *  统一状态配色（不再按角色分色）：
 *    - 运行中（running）：黄色（#ffb020）
 *    - 完成（done）：绿色（#2ee6a0）
 *    - 错误（error）：红色（#ff5c7a）
 *    - 空闲（idle）：深蓝
 * =========================================================================== */
class TopoViz {
  constructor(host) {
    this.host = host;
    this.host.innerHTML = '';
    this.svgNS = 'http://www.w3.org/2000/svg';
    this.subs = {};
    this.order = [];
    this._clickCb = null;
    this.W = 380; this.H = 360;
    this.mainXY = { x: this.W/2, y: 44 };
    this.svg = this._svg();
    this.host.appendChild(this.svg);
    const defs = document.createElementNS(this.svgNS, 'defs');
    defs.innerHTML = `
      <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>`;
    this.svg.appendChild(defs);
  }

  _svg() {
    const s = document.createElementNS(this.svgNS, 'svg');
    s.setAttribute('viewBox', `0 0 ${this.W} ${this.H}`);
    s.setAttribute('width', '100%');
    s.setAttribute('style', 'background:radial-gradient(circle at 50% 30%, #142042 0%, #070b16 80%);border-radius:8px');
    return s;
  }

  _node(x, y, r, fill, label, id, glowColor) {
    const g = document.createElementNS(this.svgNS, 'g');
    g.style.cursor = 'pointer';
    const c = document.createElementNS(this.svgNS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', r);
    c.setAttribute('fill', fill);
    c.setAttribute('stroke', glowColor || '#00d4ff');
    c.setAttribute('stroke-width', '2');
    c.setAttribute('filter', 'url(#glow)');
    c.style.transition = 'all .3s';
    const t = document.createElementNS(this.svgNS, 'text');
    t.setAttribute('x', x); t.setAttribute('y', y + r + 13);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', '9');
    t.setAttribute('fill', '#8aa6d0');
    t.textContent = label;
    g.appendChild(c); g.appendChild(t);
    if (id) g.addEventListener('click', () => this._clickCb && this._clickCb(id));
    this.svg.appendChild(g);
    return { g, c, t };
  }

  _edge(x1, y1, x2, y2, color) {
    const ln = document.createElementNS(this.svgNS, 'line');
    ln.setAttribute('x1', x1); ln.setAttribute('y1', y1);
    ln.setAttribute('x2', x2); ln.setAttribute('y2', y2);
    ln.setAttribute('stroke', color || '#1f3a6b');
    ln.setAttribute('stroke-width', '1.5');
    ln.setAttribute('stroke-dasharray', '4 4');
    this.svg.appendChild(ln);
    return ln;
  }

  setMain() {
    // 主 agent 初始状态：深蓝
    const o = this._node(this.mainXY.x, this.mainXY.y, 18, '#0a2a5e', '主 agent', 'main', '#00d4ff');
    this.subs['main'] = { ...o, x: this.mainXY.x, y: this.mainXY.y, status: 'idle' };
  }

  addSubagent(id, topic, role) {
    // 两个子分析师左右分布：看多居左，看空居右（保持空间布局可读）
    const i = this.order.length;
    const x = (role === 'bear') ? 280 : 100;
    const y = 200 + (i % 2) * 100;
    // 初始状态统一：深蓝 + 青色边
    const o = this._node(x, y, 14, '#103a5c',
      topic.length > 9 ? topic.slice(0, 9) + '…' : topic, id, '#00d4ff');
    this._edge(this.mainXY.x, this.mainXY.y + 18, x, y - 14, '#1f3a6b');
    this.subs[id] = { ...o, x, y, status: 'idle', topic, role };
    this.order.push(id);
  }

  markRunning(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'running';
    // 统一运行中：黄色
    s.c.setAttribute('stroke', '#ffb020');
    s.c.setAttribute('stroke-width', '3.5');
    s.c.setAttribute('fill', '#3a2a00');
    if (!s._pulse) {
      s._pulse = setInterval(() => {
        if (s.status !== 'running') { clearInterval(s._pulse); s._pulse = null; return; }
        s.c.setAttribute('r', s.c.getAttribute('r') === '16' ? '13' : '16');
      }, 450);
    }
  }

  markDone(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'done';
    if (s._pulse) { clearInterval(s._pulse); s._pulse = null; }
    // 统一完成：绿色
    s.c.setAttribute('fill', '#0d3d2a');
    s.c.setAttribute('stroke', '#2ee6a0');
    s.c.setAttribute('stroke-width', '2.5');
    s.c.setAttribute('r', '14');
  }

  markError(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'error';
    if (s._pulse) { clearInterval(s._pulse); s._pulse = null; }
    // 统一错误：红色
    s.c.setAttribute('fill', '#3a0d18');
    s.c.setAttribute('stroke', '#ff5c7a');
    s.c.setAttribute('stroke-width', '2.5');
    s.c.setAttribute('r', '14');
  }

  reset() {
    Object.values(this.subs).forEach(s => { if (s._pulse) clearInterval(s._pulse); });
    this.host.innerHTML = '';
    this.subs = {}; this.order = [];
  }

  onClick(cb) { this._clickCb = cb; }
}
